from typing import TypedDict, Annotated, List, Dict
import operator

def merge_lists(a: list, b: list) -> list:
    """Ensure lists merge without duplicates or None values."""
    if not a: return b or []
    if not b: return a or []
    return a + b

class AgentState(TypedDict):
    manifest_path: str
    manifest: dict
    # List of tasks parsed from the manifest
    scene_tasks: Annotated[list[dict], operator.add]
    
    # Storage for parallel outputs
    audio_tracks: Annotated[list[dict], operator.add]
    video_tracks: Annotated[list[dict], operator.add]
    
    # Post-processing outputs
    face_swapped: Annotated[list[dict], operator.add]
    final_videos: Annotated[list[dict], operator.add]
    
    error: str

def get_initial_state(manifest_path: str) -> AgentState:
    import json
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "scene_tasks": [],
        "audio_tracks": [],
        "video_tracks": [],
        "face_swapped": [],
        "final_videos": [],
        "error": ""
    }
