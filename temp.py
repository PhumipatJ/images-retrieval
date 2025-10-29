import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from DELG_Class import DELG

class ImageRetrieval:
    def __init__(self, model_class, model_path, device, use_global=True, use_local=True):
        """
        model_class: class of the model (DELG or DELF)
        model_path: path to .pth file
        device: 'cuda' or 'cpu'
        use_global: whether the model produces global descriptors
        use_local: whether the model produces local descriptors
        """
        self.device = device
        self.use_global = use_global
        self.use_local = use_local

        # Initialize model
        self.model = model_class(
            pretrained=False,
            use_global=use_global,
            use_local=use_local
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
             state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # already raw weights

        self.model.load_state_dict(state_dict)
        self.model.eval()


        # Feature storage
        self.global_features = {}  # image_path -> global descriptor
        self.local_features = {}   # image_path -> local descriptors

    @torch.no_grad()
    def extract_features(self, image_loader):
        """Extract features for all images in a dataloader."""
        for imgs, paths in tqdm(image_loader, desc="Extracting features"):
            imgs = imgs.to(self.device)
            feats = self.model(imgs)

            for i, path in enumerate(paths):
                if self.use_global and 'global' in feats:
                    self.global_features[path] = feats['global'][i].cpu().numpy()
                if self.use_local and 'local' in feats:
                    self.local_features[path] = feats['local']['descriptors'][i].cpu().numpy()

    def save_features(self, file_path):
        """Save global and local features to disk."""
        np.savez_compressed(
            file_path,
            global_features=self.global_features,
            local_features=self.local_features
        )
        print(f"Features saved to {file_path}")
        

    def load_features(self, file_path):
        """Load previously saved features."""
        data = np.load(file_path, allow_pickle=True)
        self.global_features = data['global_features'].item()
        self.local_features = data['local_features'].item()
        print(f"Features loaded from {file_path}")


    def retrieve_global(self, query_feat, top_k=5):
        """Retrieve top-k candidates using global descriptors."""
        if not self.use_global:
            raise ValueError("This model cannot produce global descriptors.")

        all_feats = np.array(list(self.global_features.values()))
        all_paths = list(self.global_features.keys())
        sims = cosine_similarity(query_feat[None, :], all_feats)[0]
        top_idx = np.argsort(-sims)[:top_k]
        return [all_paths[i] for i in top_idx], sims[top_idx]

    def rerank_local(self, query_local, candidate_paths, top_k=5):
        """Re-rank top-k candidates using local descriptors."""
        if not self.use_local:
            raise ValueError("This model cannot produce local descriptors.")

        scores = []
        for path in candidate_paths:
            target_local = self.local_features[path]
            sim_matrix = cosine_similarity(query_local, target_local)
            score = sim_matrix.max(axis=1).mean()  # max per query descriptor, then mean
            scores.append(score)

        sorted_idx = np.argsort(-np.array(scores))[:top_k]
        return [candidate_paths[i] for i in sorted_idx], np.array(scores)[sorted_idx]

    @torch.no_grad()
    def query(self, query_img, top_k=5, rerank_model=None):
        """
        query_img: torch tensor (1,3,H,W)
        top_k: number of results
        rerank_model: another ImageRetrieval instance (local descriptor for reranking)
        """
        query_img = query_img.to(self.device)
        feats = self.model(query_img)

        # Global retrieval
        if self.use_global and 'global' in feats:
            query_global = feats['global'][0].cpu().numpy()
            results, _ = self.retrieve_global(query_global, top_k=top_k)
        else:
            results = list(self.local_features.keys())[:top_k]

        # Local reranking
        if rerank_model is not None:
            query_local = feats['local']['descriptors'][0].cpu().numpy()
            results, _ = rerank_model.rerank_local(query_local, results, top_k=top_k)

        return results
