"""Historical videomae experiment model definitions.

Class bodies are preserved during extraction. Known modeling and
evaluation limitations are recorded in docs/experiment_review.md."""

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import BertModel, BertTokenizer, VideoMAEModel

from .attention import EnhancedPositionalEncoding, HandUpperBodyAttention
from .losses import LabelSmoothingLoss


class TemporalSkipConnection(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.projection = nn.Linear(feature_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vivit_features, temporal_features):
        projected_vivit = self.projection(vivit_features)
        return self.layer_norm(temporal_features + projected_vivit)


class VideoMAEEncoderWithTemporalModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name="MCG-NJU/videomae-base",
        feature_dim=768,
        hidden_dim=512,
        num_layers=2,
        temporal_model="bilstm",
        freeze_encoder=False,
        use_sliding_windows=True,
        window_size=32,
        window_stride=16,
    ):
        super().__init__()

        # Initialize VideoMAE encoder
        self.videomae = VideoMAEModel.from_pretrained(pretrained_model_name)

        if freeze_encoder:
            for param in self.videomae.parameters():
                param.requires_grad = False

        # Keep your existing attention and encoding components
        self.spatial_attention = HandUpperBodyAttention(
            input_dim=feature_dim,
            num_patches_h=14,  # For 224x224 input with 16x16 patches
            num_patches_w=14,
        )

        self.positional_encoding = EnhancedPositionalEncoding(feature_dim)

        # Configuration
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride

        # Bottleneck remains the same
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )

        # Temporal model setup - keeping your original implementation
        if temporal_model == "bilstm":
            self.temporal_model = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1,
            )
        elif temporal_model == "gru":
            self.temporal_model = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1,
            )
        elif temporal_model == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_model = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

        self.skip_connection = TemporalSkipConnection(feature_dim, hidden_dim * 2)
        self.temporal_model_type = temporal_model
        self.hidden_dim = hidden_dim * (2 if temporal_model in ["bilstm", "gru"] else 1)

    def process_window(self, video_window):
        # VideoMAE expects pixel_values as input
        outputs = self.videomae(pixel_values=video_window)
        last_hidden_state = outputs.last_hidden_state

        batch_size, seq_len, feature_dim = last_hidden_state.shape
        num_frames = video_window.size(1)
        num_patches = seq_len // num_frames

        # Reshape to match your spatial attention expectation
        truncated_len = num_frames * num_patches
        last_hidden_state = last_hidden_state[:, :truncated_len, :]
        last_hidden_state = last_hidden_state.view(
            batch_size, num_frames, num_patches, feature_dim
        )

        # Apply spatial attention and bottleneck
        frame_features = self.spatial_attention(last_hidden_state)
        compressed_features = self.bottleneck(frame_features)

        return compressed_features

    def forward(self, pixel_values):
        # Ensure input is float32
        pixel_values = pixel_values.to(dtype=torch.float32)

        if self.use_sliding_windows and pixel_values.shape[1] > self.window_size:
            frame_features = []
            for i in range(
                0, pixel_values.shape[1] - self.window_size + 1, self.window_stride
            ):
                window = pixel_values[:, i : i + self.window_size]
                features = self.process_window(window)
                frame_features.append(features)

            frame_features = torch.cat(frame_features, dim=1)
        else:
            frame_features = self.process_window(pixel_values)

        # Apply positional encoding
        frame_features = self.positional_encoding(frame_features)

        # Process through temporal model
        if self.temporal_model_type in ["bilstm", "gru"]:
            temporal_output, _ = self.temporal_model(frame_features)
        else:  # transformer
            temporal_output = self.temporal_model(frame_features)

        # Apply skip connection
        final_output = self.skip_connection(frame_features, temporal_output)

        return final_output


class SignLanguageDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim=1024,  # From your ViViT encoder output
        decoder_dim=768,  # BERT's hidden dimension
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.3,
        max_length=128,
        bert_model_name="bert-base-german-cased",
    ):
        super().__init__()

        # Initialize BERT tokenizer and model for German
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Project encoder outputs to decoder dimension
        # Add layer normalization after projection
        self.encoder_projection = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim), nn.LayerNorm(decoder_dim)
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim // 2, decoder_dim),
            nn.LayerNorm(decoder_dim),
        )

        # Positional encoding for decoder
        self.pos_encoder = nn.Embedding(max_length, decoder_dim)

        # Transformer decoder layers
        decoder_layer = TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        # Output projection
        self.output_projection = nn.Linear(decoder_dim, self.tokenizer.vocab_size)

        # Special tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.sep_token_id

    def create_mask(self, size):
        # Create causal mask for decoder
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def generate_with_beam_search(self, memory, max_length, beam_size=5):
        batch_size = memory.size(0)
        device = memory.device

        # Initialize beams: each beam contains (sequence, score)
        beams = [
            [(torch.tensor([self.bos_token_id], device=device), 0.0)]
            for _ in range(batch_size)
        ]
        completed_beams = [[] for _ in range(batch_size)]

        for _ in range(max_length - 1):
            all_candidates = []

            # Iterate through all active beams for each batch
            for i in range(batch_size):
                if len(completed_beams[i]) >= beam_size:
                    continue  # Skip processing if enough complete beams exist

                batch_candidates = []
                for sequence, score in beams[i]:
                    if sequence[-1] == self.eos_token_id:
                        # Move completed beams to final list
                        completed_beams[i].append((sequence, score))
                        continue

                    # Predict the next token probabilities
                    positions = torch.arange(sequence.size(0), device=device).unsqueeze(
                        0
                    )
                    token_embeddings = self.bert.embeddings.word_embeddings(
                        sequence.unsqueeze(0)
                    )
                    pos_embeddings = self.pos_encoder(positions)
                    decoder_input = token_embeddings + pos_embeddings
                    decoder_state = decoder_input

                    tgt_mask = self.create_mask(sequence.size(0)).to(device)
                    decoder_output = self.transformer_decoder(
                        decoder_input,
                        memory[i : i + 1],  # Memory for the specific batch
                        tgt_mask=tgt_mask,
                    )
                    logits = self.output_projection(decoder_output[:, -1])
                    probs = torch.log_softmax(logits, dim=-1)

                    # Select top-k tokens and their probabilities
                    top_values, top_indices = probs.topk(beam_size, dim=-1)
                    for token, log_prob in zip(
                        top_indices.squeeze(0), top_values.squeeze(0)
                    ):
                        new_sequence = torch.cat([sequence, token.unsqueeze(0)])
                        new_score = score + log_prob.item()
                        batch_candidates.append((new_sequence, new_score))

                # Keep top candidates for the current batch
                all_candidates.extend((i, seq, scr) for seq, scr in batch_candidates)

            # Group by batch and prune to top beam_size
            beams = [[] for _ in range(batch_size)]
            for i, seq, scr in sorted(all_candidates, key=lambda x: x[2], reverse=True):
                if len(beams[i]) < beam_size:
                    beams[i].append((seq, scr))

            # Move completed beams back
            for i in range(batch_size):
                completed_beams[i].extend(
                    [b for b in beams[i] if b[0][-1] == self.eos_token_id]
                )
                beams[i] = [b for b in beams[i] if b[0][-1] != self.eos_token_id]

        # Finalize results by picking the highest-scoring completed beam for each batch
        results = []
        for i in range(batch_size):
            if completed_beams[i]:
                results.append(
                    max(completed_beams[i], key=lambda x: x[1])[0]
                )  # Best completed beam
            else:
                results.append(
                    max(beams[i], key=lambda x: x[1])[0]
                )  # Fall back to the best incomplete beam

        # Pad results to max_length and return
        max_seq_len = max(seq.size(0) for seq in results)
        results_padded = torch.full(
            (batch_size, max_seq_len), self.pad_token_id, device=device
        )
        for i, seq in enumerate(results):
            results_padded[i, : seq.size(0)] = seq
        return results_padded

    def forward(self, encoder_output, target_tokens=None, max_length=128, beam_size=5):
        batch_size = encoder_output.size(0)
        device = encoder_output.device

        # Project encoder output
        memory = self.encoder_projection(encoder_output)
        memory = self.bottleneck(memory)

        if self.training and target_tokens is not None:
            # Training mode
            # Create position indices
            positions = torch.arange(target_tokens.size(1), device=device).expand(
                batch_size, -1
            )

            # Get token embeddings from BERT
            token_embeddings = self.bert.embeddings.word_embeddings(target_tokens)

            # Add positional encodings
            pos_embeddings = self.pos_encoder(positions)
            decoder_input = token_embeddings + pos_embeddings

            # Create causal mask
            tgt_mask = self.create_mask(target_tokens.size(1)).to(device)

            # Create padding mask
            padding_mask = target_tokens == self.pad_token_id

            # label smoothing for loss calculation
            self.criterion = LabelSmoothingLoss(
                self.tokenizer.vocab_size, smoothing=0.1
            )
            # Pass through transformer decoder
            decoder_output = self.transformer_decoder(
                decoder_input,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=padding_mask,
            )

            # Project to vocabulary
            logits = self.output_projection(decoder_output)

            return logits

        else:
            # Inference mode

            # Inference mode
            if beam_size > 1:
                # Use beam search for inference
                return self.generate_with_beam_search(
                    memory, max_length=max_length, beam_size=beam_size
                )
            else:
                # Greedy decoding for inference
                decoded_tokens = torch.full(
                    (batch_size, 1), self.bos_token_id, device=device
                )

                for _ in range(max_length - 1):
                    positions = torch.arange(
                        decoded_tokens.size(1), device=device
                    ).expand(batch_size, -1)
                    token_embeddings = self.bert.embeddings.word_embeddings(
                        decoded_tokens
                    )
                    pos_embeddings = self.pos_encoder(positions)
                    decoder_input = token_embeddings + pos_embeddings
                    decoder_state = decoder_input

                    tgt_mask = self.create_mask(decoded_tokens.size(1)).to(device)

                    decoder_output = self.transformer_decoder(
                        decoder_state, memory, tgt_mask=tgt_mask
                    )

                    # Get next token probabilities
                    next_token_logits = self.output_projection(decoder_output[:, -1:])
                    next_token = next_token_logits.argmax(dim=-1)

                    # Append to decoded tokens
                    decoded_tokens = torch.cat([decoded_tokens, next_token], dim=1)

                    # Stop if all sequences have EOS
                    if (next_token == self.eos_token_id).all():
                        break

                return decoded_tokens


class SignLanguageTranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, video, target_tokens=None):
        encoder_output = self.encoder(video)
        decoder_output = self.decoder(encoder_output, target_tokens)
        return decoder_output
