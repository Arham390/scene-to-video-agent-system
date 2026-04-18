from graph.state import AgentState
from mcp.tool_loader import loader

class SceneParser:
    def run(self, state: AgentState) -> AgentState:
        manifest = state.get("manifest", {})
        scenes = manifest.get("scenes", [])
        
        parsed_tasks = []
        for scene in scenes:
            task = loader.invoke("get_task_graph", scene=scene)
            parsed_tasks.append(task)
            
            # Commit to memory for state resumability
            loader.invoke(
                "commit_memory", 
                text=f"Scene {task['scene_id']} parsed with {len(task['audio_tasks'])} audio tasks.",
                metadata={"scene_id": task["scene_id"]}
            )
            
        print(f"[SceneParser] Parsed {len(parsed_tasks)} scenes for branch execution.")
        return {"scene_tasks": parsed_tasks}
