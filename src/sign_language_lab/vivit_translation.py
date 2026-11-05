"""Historical vivit translation model definitions.

Class bodies are preserved during extraction. Known modeling and
evaluation limitations are recorded in docs/experiment_review.md."""

import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from transformers import BertModel, BertTokenizer, VivitForVideoClassification

from .attention import EnhancedPositionalEncoding, HandUpperBodyAttention
from .losses import LabelSmoothingLoss


class TemporalSkipConnection(nn.Module):
    def __init__(self, input_dim, output_dim):  # Changed parameters here
        super().__init__()
        self.projection = nn.Linear(
            input_dim, output_dim
        )  # Projects to match temporal output
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, vivit_features, temporal_features):
        projected_vivit = self.projection(vivit_features)
        return self.layer_norm(temporal_features + projected_vivit)


class VivitEncoderWithTemporalModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name="google/vivit-b-16x2-kinetics400",
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

        # Initialize the ViViT encoder
        self.vivit = VivitForVideoClassification.from_pretrained(pretrained_model_name)
        delattr(self.vivit, "classifier")

        if freeze_encoder:
            for param in self.vivit.parameters():
                param.requires_grad = False

        self.spatial_attention = HandUpperBodyAttention(
            input_dim=feature_dim, num_patches_h=14, num_patches_w=14
        )

        self.positional_encoding = EnhancedPositionalEncoding(feature_dim)
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride

        # Bottleneck remains the same
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
        )

        # Temporal model setup with consistent dimensions
        if temporal_model in ["bilstm", "gru"]:
            temporal_hidden_dim = (
                feature_dim // 2
            )  # This ensures output_dim will be feature_dim
            if temporal_model == "bilstm":
                self.temporal_model = nn.LSTM(
                    input_size=feature_dim,
                    hidden_size=temporal_hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.1,
                )
            else:  # gru
                self.temporal_model = nn.GRU(
                    input_size=feature_dim,
                    hidden_size=temporal_hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.1,
                )
            self.output_dim = feature_dim  # Now matches BERT's hidden size
        else:  # transformer
            self.output_dim = feature_dim
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
            )
            self.temporal_model = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )

        # Skip connection with matching dimensions
        self.skip_connection = TemporalSkipConnection(
            input_dim=feature_dim,
            output_dim=feature_dim,  # Now consistently feature_dim
        )

        # Final projection to ensure output dimension matches BERT's hidden size
        self.final_projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.1),
        )

        # Configuration
        self.temporal_model_type = temporal_model
        self.feature_dim = feature_dim
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size
        self.window_stride = window_stride

    def process_window(self, video_window):
        outputs = self.vivit.vivit(video_window)
        last_hidden_state = outputs.last_hidden_state

        batch_size, seq_len, feature_dim = last_hidden_state.shape
        num_frames = video_window.size(1)
        num_patches = seq_len // num_frames

        truncated_len = num_frames * num_patches
        last_hidden_state = last_hidden_state[:, :truncated_len, :]
        last_hidden_state = last_hidden_state.view(
            batch_size, num_frames, num_patches, feature_dim
        )

        # Apply spatial attention and bottleneck
        frame_features = self.spatial_attention(last_hidden_state)
        compressed_features = self.bottleneck(frame_features)

        return compressed_features

    def forward(self, video):
        video = video.to(dtype=torch.float32)

        if self.use_sliding_windows and video.shape[1] > self.window_size:
            frame_features = []
            for i in range(
                0, video.shape[1] - self.window_size + 1, self.window_stride
            ):
                window = video[:, i : i + self.window_size]
                features = self.process_window(window)
                frame_features.append(features)

            frame_features = torch.cat(frame_features, dim=1)
        else:
            frame_features = self.process_window(video)

        # Apply positional encoding
        frame_features = self.positional_encoding(frame_features)

        # Pass through temporal model
        if self.temporal_model_type in ["bilstm", "gru"]:
            temporal_output, _ = self.temporal_model(frame_features)
        else:  # transformer
            temporal_output = self.temporal_model(frame_features)

        # Apply skip connection with matching dimensions
        skip_output = self.skip_connection(frame_features, temporal_output)

        # Final projection to ensure consistent output dimension
        final_output = self.final_projection(skip_output)

        return final_output

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SignLanguageDecoder(nn.Module):
    def __init__(
        self,
        encoder_dim=1024,  # From your ViViT encoder output
        decoder_dim=768,  # BERT's hidden dimension
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.2,
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
            nn.Linear(encoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.LayerNorm(decoder_dim),
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

    def generate_with_beam_search(
        self, memory, max_length, beam_size=5, temperature=1.0, length_penalty=1.0
    ):
        batch_size = memory.size(0)
        device = memory.device

        # Initialize beams with start tokens
        beams = [
            [
                {
                    "sequence": torch.tensor([self.bos_token_id], device=device),
                    "score": 0.0,
                    "length": 1,
                }
            ]
            for _ in range(batch_size)
        ]
        completed_beams = [[] for _ in range(batch_size)]

        for _ in range(max_length - 1):
            all_candidates = []

            for batch_idx in range(batch_size):
                if len(completed_beams[batch_idx]) >= beam_size:
                    continue

                batch_candidates = []
                for beam in beams[batch_idx]:
                    if beam["sequence"][-1] == self.eos_token_id:
                        # Add completed sequence to completed_beams
                        normalized_score = beam["score"] / (
                            beam["length"] ** length_penalty
                        )
                        completed_beams[batch_idx].append(
                            {
                                "sequence": beam["sequence"],
                                "score": normalized_score,
                                "length": beam["length"],
                            }
                        )
                        continue

                    # Prepare decoder input
                    positions = torch.arange(
                        len(beam["sequence"]), device=device
                    ).unsqueeze(0)
                    token_embeddings = self.bert.embeddings.word_embeddings(
                        beam["sequence"].unsqueeze(0)
                    )
                    pos_embeddings = self.pos_encoder(positions)
                    decoder_input = token_embeddings + pos_embeddings

                    # Create mask for decoder
                    tgt_mask = self.create_mask(len(beam["sequence"])).to(device)

                    # Get decoder output
                    decoder_output = self.transformer_decoder(
                        decoder_input,
                        memory[batch_idx : batch_idx + 1],
                        tgt_mask=tgt_mask,
                    )

                    # Get next token probabilities
                    logits = self.output_projection(decoder_output[:, -1]) / temperature
                    probs = torch.log_softmax(logits, dim=-1)

                    # Get top k candidates
                    topk_values, topk_indices = probs.topk(beam_size * 2)

                    # Add candidates
                    for token, log_prob in zip(topk_indices[0], topk_values[0]):
                        new_sequence = torch.cat([beam["sequence"], token.unsqueeze(0)])
                        new_score = beam["score"] + log_prob.item()
                        new_length = beam["length"] + 1
                        batch_candidates.append(
                            {
                                "sequence": new_sequence,
                                "score": new_score,
                                "length": new_length,
                            }
                        )

                # Sort and prune candidates
                batch_candidates = sorted(
                    batch_candidates,
                    key=lambda x: x["score"] / (x["length"] ** length_penalty),
                    reverse=True,
                )
                beams[batch_idx] = batch_candidates[:beam_size]

            # Check if all sequences are completed
            all_completed = all(
                len(completed) >= beam_size for completed in completed_beams
            )
            if all_completed:
                break

        # Finalize results
        final_sequences = []
        for batch_idx in range(batch_size):
            # Combine completed and incomplete beams
            all_beams = completed_beams[batch_idx] + beams[batch_idx]
            if not all_beams:
                # If no beams (shouldn't happen), return pad tokens
                final_sequences.append(
                    torch.full((max_length,), self.pad_token_id, device=device)
                )
                continue

            # Sort by normalized score
            sorted_beams = sorted(
                all_beams,
                key=lambda x: x["score"] / (x["length"] ** length_penalty),
                reverse=True,
            )

            # Take best sequence
            best_sequence = sorted_beams[0]["sequence"]

            # Pad if necessary
            if len(best_sequence) < max_length:
                padding = torch.full(
                    (max_length - len(best_sequence),), self.pad_token_id, device=device
                )
                best_sequence = torch.cat([best_sequence, padding])
            else:
                best_sequence = best_sequence[:max_length]

            final_sequences.append(best_sequence)

        # Stack all sequences
        return torch.stack(final_sequences)

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
        self.encoder_norm = nn.LayerNorm(encoder.output_dim)

    def forward(self, video, target_tokens=None):
        encoder_output = self.encoder(video)
        decoder_output = self.decoder(encoder_output, target_tokens)
        return decoder_output
