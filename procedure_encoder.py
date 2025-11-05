import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from scipy.sparse import csr_matrix, issparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SpecialtyConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, num_specialties, latent_dim, specialty_emb_dim=32, dropout_rate=0.3):
        super().__init__()
        
        self.specialty_embedding = nn.Embedding(num_specialties, specialty_emb_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + specialty_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + specialty_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim)
        )
    
    def encode(self, x, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        encoder_input = torch.cat([x, spec_emb], dim=1)
        return self.encoder(encoder_input)
    
    def decode(self, z, specialty_id):
        spec_emb = self.specialty_embedding(specialty_id)
        decoder_input = torch.cat([z, spec_emb], dim=1)
        return self.decoder(decoder_input)
    
    def forward(self, x, specialty_id):
        z = self.encode(x, specialty_id)
        recon = self.decode(z, specialty_id)
        return recon, z


class ReductionAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class CompressionNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        return self.network(x)


class ProcedureEmbeddingEncoder:
    """
    Inference wrapper for encoding providers through the entire pipeline.
    
    Usage:
        encoder = ProcedureEmbeddingEncoder(model_dir='./models')
        
        # Encode new provider
        new_provider_vector = load_procedure_vector()  # [num_codes]
        embeddings = encoder.encode_full_pipeline(new_provider_vector)
        
        # Access at any stage
        print(embeddings['agnostic'].shape)    # [1, latent_dim]
        print(embeddings['multiview'].shape)   # [1, 15 × latent_dim]
        print(embeddings['reduced'].shape)     # [1, reduction_dim]
        print(embeddings['final'].shape)       # [1, 128]
    """
    
    def __init__(self, model_dir='./'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Initializing encoder on device: {self.device}")
        
        with open(f'{model_dir}/final_pipeline_metadata.pkl', 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.phase1_latent_dim = self.metadata['phase1_latent_dim']
        self.reduction_dim = self.metadata['phase3a_reduction_dim']
        self.final_dim = self.metadata['phase3b_final_dim']
        self.num_specialties = self.metadata['num_specialties']
        self.specialty_code_indices = self.metadata['specialty_code_indices']
        self.input_dim = len(self.specialty_code_indices)
        self.specialty_to_id = self.metadata['specialty_to_id']
        self.id_to_specialty = {v: k for k, v in self.specialty_to_id.items()}
        
        print(f"\nPipeline configuration:")
        print(f"  Phase 1 latent dim: {self.phase1_latent_dim}")
        print(f"  Num specialties: {self.num_specialties}")
        print(f"  Input dim (filtered): {self.input_dim}")
        print(f"  Multiview dim: {self.num_specialties * self.phase1_latent_dim}")
        print(f"  Reduction dim: {self.reduction_dim}")
        print(f"  Final dim: {self.final_dim}")
        
        print("\nLoading Phase 1 model...")
        self.phase1_model = SpecialtyConditionedAutoencoder(
            self.input_dim, 
            self.num_specialties, 
            self.phase1_latent_dim
        ).to(self.device)
        self.phase1_model.load_state_dict(
            torch.load(f'{model_dir}/phase1_specialty_conditioned_autoencoder.pth', 
                      map_location=self.device)
        )
        self.phase1_model.eval()
        
        print("Loading Phase 3A model...")
        multiview_dim = self.num_specialties * self.phase1_latent_dim
        self.phase3a_model = ReductionAutoencoder(
            multiview_dim, 
            self.reduction_dim
        ).to(self.device)
        self.phase3a_model.load_state_dict(
            torch.load(f'{model_dir}/phase3a_reduction_autoencoder.pth', 
                      map_location=self.device)
        )
        self.phase3a_model.eval()
        
        print("Loading Phase 3B model...")
        self.phase3b_model = CompressionNetwork(
            self.reduction_dim, 
            self.final_dim
        ).to(self.device)
        self.phase3b_model.load_state_dict(
            torch.load(f'{model_dir}/phase3b_compression_network.pth', 
                      map_location=self.device)
        )
        self.phase3b_model.eval()
        
        print("\nEncoder initialized successfully!")
    
    def preprocess_input(self, procedure_vector):
        """Filter to specialty-relevant codes and convert to tensor"""
        if issparse(procedure_vector):
            procedure_vector = procedure_vector.toarray().flatten()
        
        if isinstance(procedure_vector, torch.Tensor):
            procedure_vector = procedure_vector.cpu().numpy()
        
        filtered = procedure_vector[self.specialty_code_indices]
        return torch.FloatTensor(filtered).unsqueeze(0).to(self.device)
    
    def encode_phase1(self, procedure_vector, specialty_id):
        """
        Encode with specific specialty view.
        
        Args:
            procedure_vector: Raw procedure vector [num_codes] or sparse matrix
            specialty_id: Specialty ID (0 to num_specialties-1) or specialty name
        
        Returns:
            embedding: numpy array [1, latent_dim]
        """
        if isinstance(specialty_id, str):
            specialty_id = self.specialty_to_id[specialty_id]
        
        x = self.preprocess_input(procedure_vector)
        spec_id_tensor = torch.LongTensor([specialty_id]).to(self.device)
        
        with torch.no_grad():
            embedding = self.phase1_model.encode(x, spec_id_tensor)
        
        return embedding.cpu().numpy()
    
    def encode_specialty_agnostic(self, procedure_vector):
        """
        Encode with specialty-agnostic view (average across all specialties).
        
        Args:
            procedure_vector: Raw procedure vector [num_codes] or sparse matrix
        
        Returns:
            embedding: numpy array [1, latent_dim]
        """
        x = self.preprocess_input(procedure_vector)
        
        all_embeddings = []
        with torch.no_grad():
            for spec_id in range(self.num_specialties):
                spec_id_tensor = torch.full((1,), spec_id, dtype=torch.long, device=self.device)
                embedding = self.phase1_model.encode(x, spec_id_tensor)
                all_embeddings.append(embedding.cpu().numpy())
        
        avg_embedding = np.mean(all_embeddings, axis=0)
        return avg_embedding
    
    def encode_multiview(self, procedure_vector):
        """
        Encode with all specialty views concatenated.
        
        Args:
            procedure_vector: Raw procedure vector [num_codes] or sparse matrix
        
        Returns:
            embedding: numpy array [1, num_specialties × latent_dim]
        """
        x = self.preprocess_input(procedure_vector)
        
        all_views = []
        with torch.no_grad():
            for spec_id in range(self.num_specialties):
                spec_id_tensor = torch.full((1,), spec_id, dtype=torch.long, device=self.device)
                embedding = self.phase1_model.encode(x, spec_id_tensor)
                all_views.append(embedding.cpu().numpy())
        
        multiview = np.concatenate(all_views, axis=1)
        return multiview
    
    def encode_reduced(self, multiview_embedding):
        """
        Reduce multiview embedding with Phase 3A.
        
        Args:
            multiview_embedding: Multi-view embedding [1, num_specialties × latent_dim]
        
        Returns:
            embedding: numpy array [1, reduction_dim]
        """
        x = torch.FloatTensor(multiview_embedding).to(self.device)
        
        with torch.no_grad():
            _, reduced = self.phase3a_model(x)
        
        return reduced.cpu().numpy()
    
    def encode_final(self, reduced_embedding):
        """
        Final compression with Phase 3B.
        
        Args:
            reduced_embedding: Reduced embedding [1, reduction_dim]
        
        Returns:
            embedding: numpy array [1, final_dim] (L2 normalized)
        """
        x = torch.FloatTensor(reduced_embedding).to(self.device)
        
        with torch.no_grad():
            final = self.phase3b_model(x)
            final_norm = F.normalize(final, p=2, dim=1)
        
        return final_norm.cpu().numpy()
    
    def encode_full_pipeline(self, procedure_vector, return_intermediates=True):
        """
        Encode through entire pipeline.
        
        Args:
            procedure_vector: Raw procedure vector [num_codes] or sparse matrix
            return_intermediates: If True, return dict with all stages
        
        Returns:
            If return_intermediates=True:
                dict with keys: 'agnostic', 'multiview', 'reduced', 'final'
            If return_intermediates=False:
                final embedding [1, final_dim]
        """
        agnostic = self.encode_specialty_agnostic(procedure_vector)
        multiview = self.encode_multiview(procedure_vector)
        reduced = self.encode_reduced(multiview)
        final = self.encode_final(reduced)
        
        if return_intermediates:
            return {
                'agnostic': agnostic,
                'multiview': multiview,
                'reduced': reduced,
                'final': final
            }
        else:
            return final
    
    def get_specialty_names(self):
        """Get list of specialty names in order"""
        return [self.id_to_specialty[i] for i in range(self.num_specialties)]


if __name__ == '__main__':
    print("Testing encoder...")
    encoder = ProcedureEmbeddingEncoder(model_dir='.')
    
    dummy_vector = np.random.rand(10000)
    embeddings = encoder.encode_full_pipeline(dummy_vector)
    
    print("\nTest encoding successful!")
    print(f"  Agnostic: {embeddings['agnostic'].shape}")
    print(f"  Multiview: {embeddings['multiview'].shape}")
    print(f"  Reduced: {embeddings['reduced'].shape}")
    print(f"  Final: {embeddings['final'].shape}")
