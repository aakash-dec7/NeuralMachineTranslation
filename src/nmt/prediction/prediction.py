import torch
from main import model, tokenizer
from src.nmt.logger import logger
from src.nmt.entity.entity import PredictionConfig


class Prediction:
    def __init__(self, config: PredictionConfig):
        """Initializes the Prediction class with configuration settings."""
        self.config: PredictionConfig = config
        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model
        self.tokenizer = tokenizer

    def _preprocess_input(self, input_text: str) -> torch.Tensor:
        """Tokenizes and pads the input text for model inference."""
        try:
            input_text = input_text.strip()
            if not input_text:
                logger.warning("Received empty input text.")
                return torch.zeros((1, self.config.params.max_length), dtype=torch.long)

            tokenized_input = self.tokenizer(
                [input_text],
                padding=True,
                truncation=True,
                max_length=self.config.params.max_length,
                return_tensors="pt",
            ).input_ids

            return tokenized_input
        except Exception as e:
            logger.exception(f"Error during input preprocessing: {e}")
            return torch.zeros((1, self.config.params.max_length), dtype=torch.long)

    def _encode_input(
        self, input_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Passes input tensor through the model encoder."""
        try:
            input_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)
            embedded = self.model.embedding(input_tensor)
            positional_encoded = self.model.positional_encoding(embedded)

            for layer in self.model.encoder_layers:
                positional_encoded = layer(positional_encoded, input_mask)

            return positional_encoded, input_mask
        except Exception as e:
            logger.exception(f"Error during encoding: {e}")
            return torch.zeros_like(input_tensor), torch.zeros_like(input_tensor)

    def _decode_sequence(
        self, encoder_output: torch.Tensor, input_mask: torch.Tensor
    ) -> str:
        """Decodes the output sequence from the model using greedy decoding."""
        try:
            bos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.bos_token
            )
            eos_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.eos_token
            )
            pad_token: int = self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.pad_token
            )

            if bos_token is None or eos_token is None:
                raise ValueError("Tokenizer is missing required BOS or EOS tokens.")

            x_input = torch.tensor([[bos_token]], dtype=torch.long)
            translated_tokens = []

            for _ in range(self.config.params.max_length):
                with torch.no_grad():
                    embedded_target = self.model.embedding(x_input)
                    target_positional = self.model.positional_encoding(embedded_target)
                    target_mask = torch.triu(
                        torch.ones(1, x_input.size(1), x_input.size(1)), diagonal=1
                    ).bool()

                    decoder_output = target_positional
                    for layer in self.model.decoder_layers:
                        decoder_output = layer(
                            decoder_output, encoder_output, input_mask, target_mask
                        )

                    predicted_token = torch.argmax(
                        self.model.fc_out(decoder_output[:, -1, :]), dim=-1
                    ).item()

                if predicted_token in {eos_token, pad_token}:
                    break

                translated_tokens.append(predicted_token)
                x_input = torch.cat(
                    [x_input, torch.tensor([[predicted_token]], dtype=torch.long)],
                    dim=1,
                )

            return self.tokenizer.decode(translated_tokens)
        except Exception as e:
            logger.exception(f"Error during sequence decoding: {e}")
            return ""

    def predict(self, input_text: str) -> str:
        """Generates a prediction based on the given input text."""
        try:
            logger.info(f"Processing input text: {input_text}")
            input_tensor = self._preprocess_input(input_text)
            encoder_output, input_mask = self._encode_input(input_tensor)
            translated_text = self._decode_sequence(encoder_output, input_mask)
            logger.info(f"Prediction completed successfully: {translated_text}")
            return translated_text
        except Exception as e:
            logger.exception(f"Prediction error: {e}")
            return ""
