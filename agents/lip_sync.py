import os
from graph.state import AgentState
from mcp.tool_loader import loader
from config import config

class LipSync:
    def run(self, state: AgentState) -> dict:
        """
        Gathers all generated audio and video outputs from parallel branches
        and synchronizes them into the final scene outputs.
        """
        audios = state.get("audio_tracks", [])
        videos = state.get("face_swapped", [])
        
        final_videos = []
        
        # Match by scene_id
        for v in videos:
            scene_id = v.get("scene_id")
            video_file = v.get("video_file")
            
            # Find matching audio
            matching_audio = None
            for a in audios:
                if a.get("scene_id") == scene_id:
                    # In a real system, you'd merge all audio files into one sequence.
                    # We just take the first one or pass the directory for stubbing.
                    audio_files = a.get("audio_files", [])
                    if audio_files:
                        matching_audio = audio_files[0]
                    break
            
            out_file = os.path.join(config.raw_scenes_dir, f"scene_{scene_id}_final.mp4")
            
            if matching_audio and video_file:
                merged = loader.invoke(
                    "lip_sync_aligner",
                    audio_path=matching_audio,
                    video_path=video_file,
                    output_path=out_file
                )
                final_videos.append({"scene_id": scene_id, "video_file": merged})
                print(f"[LipSync] Synchronized Scene {scene_id}.")
            else:
                print(f"[LipSync] Warning: Missing audio or video for Scene {scene_id}.")
                
        return {"final_videos": final_videos}
