import os
from mcp.tool_loader import loader
from config import config

class FaceSwap:
    def run(self, video_track: dict) -> dict:
        """
        Maps characters onto the video track.
        """
        scene_id = video_track.get("scene_id")
        base_video = video_track.get("video_file")
        
        out_file = os.path.join(config.raw_scenes_dir, f"scene_{scene_id}_faceswapped.png")
        
        # Call swapper
        swapped_file = loader.invoke(
            "face_swapper",
            input_video=base_video,
            character_id="mixed_cast",
            output_path=out_file
        )
        
        # Call Validator
        is_valid = loader.invoke("identity_validator", mapped_video=swapped_file)
        if not is_valid:
            print(f"[FaceSwap] Warning: Identity validation failed for Scene {scene_id}")
            
        print(f"[FaceSwap] Applied character faces for Scene {scene_id}.")
        return {"face_swapped": [{"scene_id": scene_id, "video_file": swapped_file}]}
