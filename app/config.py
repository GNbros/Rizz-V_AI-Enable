from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -----------------------------------------------------------------------
    # Model — only these two lines need to change after each retrain
    # -----------------------------------------------------------------------
    base_model_name: str = "Salesforce/codegen-350M-multi"
    adapter_path: str = "trained_model/final_model"

    # Human-readable label so GET / always shows which model is live
    model_version: str = "v1"

    # -----------------------------------------------------------------------
    # FIM tokens — change if retraining with a different tokenizer
    # -----------------------------------------------------------------------
    fim_prefix: str = "<fim_prefix>"
    fim_suffix: str = "<fim_suffix>"
    fim_middle: str = "<fim_middle>"

    # -----------------------------------------------------------------------
    # Inference limits
    # -----------------------------------------------------------------------
    max_input_length: int = 512
    default_max_new_tokens: int = 30

    # -----------------------------------------------------------------------
    # Database
    # -----------------------------------------------------------------------
    db_path: str = "ratings.db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
