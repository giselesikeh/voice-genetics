
from typing import Dict


def get_privacy_status() -> Dict[str, bool]:
    """
    Privacy status for the current MVP.

    Raw audio is not intentionally stored permanently.
    Temporary files are created only during request processing and deleted immediately.
    """
    return {
        "raw_audio_stored": False,
        "temporary_files_deleted": True,
        "output_contains_raw_audio": False,
    }
