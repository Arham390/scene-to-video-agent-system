import os
from graph.state import AgentState
from mcp.tool_loader import loader
from config import config

class VideoGen:
    def run(self, task: dict) -> dict:
        """
        Generates video sequence approximations for a single scene task.
        """
        scene_id = task.get("scene_id")
        video_tasks = task.get("video_tasks", [])
        
        out_file = os.path.join(config.raw_scenes_dir, f"scene_{scene_id}_video_base.png")
        
        result_path = loader.invoke(
            "query_stock_footage",
            visual_cues=video_tasks,
            output_path=out_file
        )
        
        print(f"[VideoGen] Generated base video frames for Scene {scene_id}.")
        return {"video_tracks": [{"scene_id": scene_id, "video_file": result_path}]}
