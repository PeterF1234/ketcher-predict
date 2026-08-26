from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rdkit import Chem
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
from torch_geometric.data import Data
from torch_geometric.nn.models import AttentiveFP
import torch.nn.functional as F
from torch_geometric.nn import GPSConv, GraphNorm, global_mean_pool, GINConv
from torch_geometric.utils import get_laplacian, to_dense_adj
from pathlib import Path
from torch_geometric.data import Batch
from fastapi.responses import FileResponse

# Make the paths relative to the Python file, not the shell location.
BASE_DIR = Path(__file__).resolve().parent
PARAM_FILE_A = BASE_DIR / "Model6b_optparams.json"
WEIGHTS_FILE_A = BASE_DIR / "model6b_trial_8.pt"
PARAM_FILE_B = BASE_DIR / "Model2_optparams.json"
WEIGHTS_FILE_B = BASE_DIR / "model2_trial_2.pt"
HTML_PATH = BASE_DIR.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"running on {device}")

# Global placeholders
tokenizer = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model_6b, model_2

    #-------------------------
    # Model_6b
    #-------------------------

    with open(PARAM_FILE_A, "r") as f:
        optparams = json.load(f)
    
    graph_hidden_dim = optparams["graph_hidden_dim"]
    num_graph_layers = optparams["num_graph_layers"]
    graph_activation = optparams["graph_activation"]
    graph_dropout = optparams["graph_dropout"]
    use_fusion = optparams["use_fusion"]
    fusion_dim = optparams["fusion_dim"]
    pooling = optparams["pooling"]
    head_activation = optparams["head_activation"]
    head_dim = optparams["head_dim"]

    # Load ONCE at startup
    tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k", clean_up_tokenization_spaces=False)
    model_6b = ChemBERTaGraphHybrid(chemberta_name="seyonec/PubChem10M_SMILES_BPE_450k",
                                    node_feat_dim=6,      # must match Data.x size (= input dimension for the graph encoder)
                                    graph_hidden_dim=graph_hidden_dim,
                                    num_graph_layers=num_graph_layers,
                                    fusion_dim=fusion_dim,
                                    graph_activation=graph_activation,
                                    graph_dropout=graph_dropout,
                                    head_activation=head_activation,
                                    head_dim=head_dim,
                                    use_fusion=use_fusion,
                                    pooling=pooling).to(device)
    model_6b.load_state_dict(torch.load(WEIGHTS_FILE_A, map_location=device))
    model_6b.eval()

    print("Model_6b loaded.")

    #-------------------------
    # Model_2
    #-------------------------

    with open(PARAM_FILE_B, "r") as f:
        optparams_ = json.load(f)

    hidden_channels = optparams_['hidden_channels']
    num_layers = optparams_['num_layers']
    num_timesteps = optparams_['num_timesteps']
    dropout = optparams_['dropout']
    # these two are fixed through the smiles_to_graph() function
    node_feat_dim = 10 # data.x.shape[1]
    edge_feat_dim = 8  # data.edge_attr.shape[1]

    model_2 = AttentiveGraphRegressor(node_feat_dim=node_feat_dim,
                                      edge_feat_dim=edge_feat_dim,
                                      hidden_channels=hidden_channels,
                                      num_layers=num_layers,
                                      num_timesteps=num_timesteps,
                                      dropout=dropout).to(device)
    model_2.load_state_dict(torch.load(WEIGHTS_FILE_B, map_location=device))
    model_2.eval()

    print("Model_2 loaded.")

    yield  # <-- FastAPI starts serving here

    # Optional cleanup on shutdown
    print("Shutting down…")
    del model_6b
    del model_2
    del tokenizer

app = FastAPI(lifespan=lifespan)
# app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # allow all origins
    allow_credentials=True,
    allow_methods=["*"],      # allow all HTTP methods
    allow_headers=["*"],      # allow all headers
)

class MolRequest(BaseModel):
    smiles: str
    model: str

# -------------------------
#
# Model_6b Definition
#
# -------------------------

# -------------------------
# Atom feature extraction
# -------------------------
def atom_features(atom):
    return torch.tensor([
        atom.GetAtomicNum(),                 # atomic number
        atom.GetTotalDegree(),               # degree
        atom.GetFormalCharge(),              # formal charge
        int(atom.GetIsAromatic()),           # aromaticity
        atom.GetTotalNumHs(),                # number of Hs
        atom.GetHybridization().real,        # hybridization enum
    ], dtype=torch.float)

# -------------------------
# Bond feature extraction
# -------------------------
def bond_features(bond):
    bt = bond.GetBondType()
    return torch.tensor([
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
    ], dtype=torch.float)

# -------------------------
# Convert SMILES → PyG Data
# -------------------------
def smiles_to_pyg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features
    x = torch.stack([atom_features(atom) for atom in mol.GetAtoms()], dim=0)

    # Edges
    edge_index = []
    edge_attr = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        bf = bond_features(bond)

        # Undirected graph → add both directions
        edge_index.append([i, j])
        edge_attr.append(bf)

        edge_index.append([j, i])
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.stack(edge_attr, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
# ---------------------------------------------------------
# Laplacian Positional Encoding (LapPE)
# ---------------------------------------------------------
def compute_lappe(edge_index, num_nodes, k, x):
    edge_index, edge_weight = get_laplacian(edge_index, normalization="sym", num_nodes=num_nodes)
    L = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0].to(x.device)

    eigvals, eigvecs = torch.linalg.eigh(L)
    return eigvecs[:, :k]

# ---------------------------------------------------------
# Random Walk Structural Encoding (RWSE)
# ---------------------------------------------------------
def compute_rwse(edge_index, num_nodes, steps, x):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(x.device)
    deg = A.sum(dim=-1, keepdim=True) + 1e-8
    P = A / deg

    P_t = torch.eye(num_nodes, device=x.device, dtype=x.dtype)
    rwse_list = []

    for _ in range(steps):
        P_t = P_t @ P
        rwse_list.append(torch.diagonal(P_t).unsqueeze(-1))

    return torch.cat(rwse_list, dim=-1)

# ---------------------------------------------------------
# Centrality Encoding (degree + eigenvector centrality)
# ---------------------------------------------------------
def compute_centrality(edge_index, num_nodes, x):
    A = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].to(x.device)

    deg = A.sum(dim=-1, keepdim=True)

    v = torch.rand(num_nodes, 1, device=x.device, dtype=x.dtype)
    for _ in range(20):
        v = A @ v
        v = v / (v.norm() + 1e-8)

    return torch.cat([deg, v], dim=-1)

# ---------------------------------------------------------
# Add padding for short SMILES 
# (We set lappe_dim=8 in GPSE_GPSConvEncoder, 
# so compute_lappe() will give wrong dimension 
# for graphs with fewer than 8 nodes
# that make input_proj(x) fail in the forward pass.
# Same can be true for compute_rwse())
# ---------------------------------------------------------

def pad_to_dim(tensor, dim):
    # tensor: [num_nodes, d]
    d = tensor.size(1)
    if d < dim:
        pad = dim - d
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    return tensor

# ---------------------------------------------------------
# GPSE + GPSConv Encoder
# ---------------------------------------------------------
def make_gps_layer(hidden_dim):
    # Local GNN: GIN
    local_gnn = GINConv(
        nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    )

    # GPSConv with PyG 2.7.0 API
    return GPSConv(
        channels=hidden_dim,
        conv=local_gnn,
        attn_type='multihead',
        heads=4,
        dropout=0.1, # we do not optimize this
        act='relu'
    )

class GPSE_GPSConvEncoder(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        hidden_dim=128,
        num_layers=4,
        graph_activation="gelu", # "relu", "gelu", "silu"
        dropout=0.1,
        rwse_steps=8,
        lappe_dim=8,
        centrality_dim=2,
    ):
        super().__init__()

        self.graph_activation = graph_activation
        self.rwse_steps = rwse_steps
        self.lappe_dim = lappe_dim
        self.centrality_dim = centrality_dim

        total_struct_dim = rwse_steps + lappe_dim + centrality_dim

        self.input_proj = nn.Linear(node_feat_dim + total_struct_dim, hidden_dim)
        self.layers = nn.ModuleList([make_gps_layer(hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch): # compatible with Model 5
        num_nodes = x.size(0)

        rwse = compute_rwse(edge_index, num_nodes, self.rwse_steps, x)
        lappe = compute_lappe(edge_index, num_nodes, self.lappe_dim, x)
        cent = compute_centrality(edge_index, num_nodes, x)

        # print("x:", x.shape)
        # print("rwse:", rwse.shape)
        # print("lappe:", lappe.shape)
        # print("cent:", cent.shape)

        rwse  = pad_to_dim(rwse, self.rwse_steps)
        lappe = pad_to_dim(lappe, self.lappe_dim)
        cent  = pad_to_dim(cent, self.centrality_dim)
        x = torch.cat([x, rwse, lappe, cent], dim=-1)
        x = self.input_proj(x)

        for conv, norm in zip(self.layers, self.norms):
            x = conv(x, edge_index, batch=batch)
            x = norm(x, batch)
            if self.graph_activation == "relu":
                x = F.relu(x)
            elif self.graph_activation == "gelu":
                x = F.gelu(x)
            elif self.graph_activation == "silu":
                x = F.silu(x)
            x = self.dropout(x)

        # Because GPSConv mixes local and global attention, 
        # the node embeddings are already normalized in a way that makes mean pooling a natural fit.
        return global_mean_pool(x, batch)

#----------------------
# ChemBERTa + PyG hybrid model
#----------------------

class GatedFusion(nn.Module):
    def __init__(self, chem_dim, graph_dim, hidden_dim):
        super().__init__()

        # Project both modalities to a shared hidden space
        self.proj_chem = nn.Linear(chem_dim, hidden_dim)
        self.proj_graph = nn.Linear(graph_dim, hidden_dim)

        # Gating network: decides how much to trust each modality
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, chem_repr, graph_repr):
        # Project to shared space
        c = self.proj_chem(chem_repr)
        g = self.proj_graph(graph_repr)

        # Compute gate value α in [0, 1]
        alpha = self.gate(torch.cat([c, g], dim=-1))

        # Gated combination
        fused = alpha * c + (1 - alpha) * g
        return fused

class ChemBERTaGraphHybrid(nn.Module):
    def __init__(
        self,
        chemberta_name: str = "seyonec/PubChem10M_SMILES_BPE_450k",
        node_feat_dim: int = 32,
        graph_hidden_dim: int = 256,
        fusion_dim: int = 512,
        num_graph_layers: int = 5,
        graph_activation="gelu",
        graph_dropout=0.1,
        # out_dim: int = 1,
        freeze_chemberta: bool = False,
        freeze_graph: bool = False,
        use_fusion=True,
        head_activation="gelu",
        head_dim: int = 512,
        pooling="cls" # "mean" or "cls"
    ):
        super().__init__()

        self.pooling = pooling
        self.use_fusion = use_fusion
        # ChemBERTa encoder
        self.chemberta = AutoModel.from_pretrained(chemberta_name)
        self.chem_hidden_dim = self.chemberta.config.hidden_size
        self.graph_hidden_dim = graph_hidden_dim
        self.fusion_dim = fusion_dim

        if freeze_chemberta:
            for p in self.chemberta.parameters():
                p.requires_grad = False

        # GPSE + GPSConv Encoder
        self.graph_encoder = GPSE_GPSConvEncoder(
            node_feat_dim=node_feat_dim,
            hidden_dim=self.graph_hidden_dim,
            num_layers=num_graph_layers,
            graph_activation=graph_activation,
            dropout=graph_dropout,
            rwse_steps=8,
            lappe_dim=8,
            centrality_dim=2
        )

        if freeze_graph:
            for p in self.graph_encoder.parameters():
                p.requires_grad = False

        # Optional GatedFusion
        if self.use_fusion:
            self.fusion = GatedFusion(chem_dim=self.chem_hidden_dim, graph_dim=self.graph_hidden_dim, hidden_dim=self.fusion_dim)
            fused_dim = self.fusion_dim
        else:
            # No fusion → concatenate raw representations
            fused_dim = self.chem_hidden_dim + self.graph_hidden_dim

        # Prediction head
        act = nn.GELU() if head_activation == "gelu" else nn.SiLU()
        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim), 
            nn.Linear(fused_dim, head_dim), # 2x expansion might not be optimal for hybrid models
            act,
            nn.Linear(head_dim, 1), # dropout not recommended for a head this small
        )

    def forward(self, chem_inputs, pyg_batch):
        # chem_inputs: dict from ChemBERTa tokenizer
        chem_out = self.chemberta(**chem_inputs)
        # chem_repr = chem_out.last_hidden_state[:, 0]  # [B, chem_hidden_dim]

        if self.pooling == "cls":
            chem_repr = chem_out.last_hidden_state[:, 0]
        else:
            mask = chem_inputs["attention_mask"].unsqueeze(-1)
            chem_repr = (chem_out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        # pyg_batch: a torch_geometric.data.Batch
        x, edge_index, batch = pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch
        graph_repr = self.graph_encoder(x, edge_index, batch)  # [B, graph_hidden_dim]

        if self.use_fusion:
            fused = self.fusion(chem_repr, graph_repr) # [B, fusion_dim]
        else:
            fused = torch.cat([chem_repr, graph_repr], dim=-1)

        pred = self.head(fused)
        return pred

# -------------------------
#
# Model_2 Definition
#
# -------------------------

HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

def one_hot(value, choices):
    return [int(value == choice) for choice in choices]

def atom_features_(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        *one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES), # Better to do this way, hybridization is a categorical variable, not a continuous one. One‑hot encoding preserves that structure.
    ]

def bond_features_(bond):
    bt = bond.GetBondType()
    stereo = bond.GetStereo()
    return [
        int(bt == Chem.rdchem.BondType.SINGLE),
        int(bt == Chem.rdchem.BondType.DOUBLE),
        int(bt == Chem.rdchem.BondType.TRIPLE),
        int(bt == Chem.rdchem.BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(stereo == Chem.rdchem.BondStereo.STEREOZ),
        int(stereo == Chem.rdchem.BondStereo.STEREOE),
    ]

def smiles_to_graph(smiles, y=None):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')

    x = torch.tensor([atom_features_(atom) for atom in mol.GetAtoms()], dtype=torch.float32)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features_(bond)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 8), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=str(smiles))
    if y is not None:
        data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data

class MolGraphDataset(Dataset):
    def __init__(self, smiles_list, y_list):
        super().__init__()
        assert len(smiles_list) == len(y_list), 'The lengths of smiles and y values are not equal!'
        self.graphs = [smiles_to_graph(smiles, y) for smiles, y in zip(smiles_list, y_list)]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

class AttentiveGraphRegressor(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_channels, num_layers, num_timesteps, dropout):
        super().__init__()
        self.model = AttentiveFP(
            in_channels=node_feat_dim,
            hidden_channels=hidden_channels,
            out_channels=1,
            edge_dim=edge_feat_dim,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

    def forward(self, data):
        return self.model(data.x, data.edge_index, data.edge_attr, data.batch).view(-1)


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if (mol is None) or (smiles == ""): # invalid SMILES
        raise ValueError("Invalid SMILES!")
    n_fragments = len(Chem.GetMolFrags(mol))
    if n_fragments > 1:
        raise ValueError("Multiple structures!")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("R groups are not supported!")
    return Chem.MolToSmiles(mol, canonical=True)

# def get_predval(smiles, model, device="cpu"): # single smiles prediction
#     model.eval()
#     smiles = canonicalize(smiles)
#     chem_input = tokenizer(
#         [smiles],
#         padding=True,
#         truncation=True,
#         return_tensors="pt",
#         max_length=64)
#     chem_input = {k: v.to(device) for k, v in chem_input.items()}
#     graph_input = smiles_to_pyg(smiles)
#     graph_input = graph_input.to(device)
#     with torch.no_grad():
#         prediction = model(chem_input, graph_input)
#     prediction = prediction.detach().cpu().numpy()
#     return round(prediction.item(), 4)

# -------------------------
#
# Using the Model
#
# -------------------------

# @app.post("/predict")
# def get_prediction(req: MolRequest):

#     print(f"input SMILES: {req.smiles}")
#     predval = get_predval(req.smiles, model)
#     return predval

@app.post("/predict")
def get_prediction(req: MolRequest):

    print(f"input SMILES: {req.smiles}")
    print(f"model chosen: {req.model}")
    try:
        smiles = canonicalize(req.smiles)
    except ValueError as e:
        return {"prediction": str(e)}
    if req.model == "Model_6b":
        chem_input = tokenizer(
            [smiles],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=64)
        chem_input = {k: v.to(device) for k, v in chem_input.items()}
        graph_input = smiles_to_pyg(smiles)
        graph_input = graph_input.to(device)
        with torch.no_grad():
            prediction = model_6b(chem_input, graph_input)
        prediction = prediction.detach().cpu().numpy()
        prediction = round(prediction.item(), 3)
        prediction = f"{prediction} eV"
        print(f"prediction: {prediction}")
        return {"prediction": prediction}
    elif req.model == "Model_2":
        dta = smiles_to_graph(smiles).to(device)
        with torch.no_grad():
            batch = Batch.from_data_list([dta])
            pred = model_2(batch)
        prediction = pred.detach().cpu().numpy()
        prediction = round(prediction.item(), 3)
        prediction = f"{prediction} eV"
        print(f"prediction: {prediction}")
        return {"prediction": prediction}
    else:
        raise HTTPException(400, detail="Model selection invalid!")

# @app.post("/predict")
# def get_prediction(req: MolRequest): # smi_to_pred = ["C1=CC=C1", "N#CC1=C(N)C(C#N)=C1N"]

#     return  {"prediction": req.smiles}

@app.get("/")
async def serve_index(): # avoid serving an older cached index.html
    return FileResponse("/app/index.html", headers={"Cache-Control": "no-cache"})

app.mount("/static", StaticFiles(directory="/app/static"), name="static")
# app.mount("/", StaticFiles(directory=HTML_PATH, html=True), name="static")

@app.get("/manifest.json")
async def manifest():
    return FileResponse("/app/manifest.json")

@app.get("/asset-manifest.json")
async def manifest():
    return FileResponse("/app/asset-manifest.json")

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("/app/favicon.ico")

@app.get("/favicon-16x16.png")
async def favicon():
    return FileResponse("/app/favicon-16x16.png")

@app.get("/favicon-32x32.png")
async def favicon():
    return FileResponse("/app/favicon-32x32.png")

@app.get("/apple-touch-icon.png")
async def favicon():
    return FileResponse("/app/apple-touch-icon.png")

@app.get("/indigo-ketcher-1.27.0.wasm")
async def favicon():
    return FileResponse("/app/indigo-ketcher-1.27.0.wasm")

@app.get("/indigo-ketcher-norender-1.27.0.wasm")
async def favicon():
    return FileResponse("/app/indigo-ketcher-norender-1.27.0.wasm")
