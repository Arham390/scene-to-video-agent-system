from langgraph.graph import StateGraph, END
try:
    from langgraph.constants import Send
except ImportError:
    # Older langgraph (<0.1.0) might have it in `langgraph.graph` or custom routing
    from langgraph.graph import Send

from graph.state import AgentState
from agents.scene_parser import SceneParser
from agents.voice_synth import VoiceSynth
from agents.video_gen import VideoGen
from agents.face_swap import FaceSwap
from agents.lip_sync import LipSync

# Instantiate Agents
_scene_parser = SceneParser()
_voice_synth = VoiceSynth()
_video_gen = VideoGen()
_face_swap = FaceSwap()
_lip_sync = LipSync()

# Node wrappers
def scene_parser_node(state: AgentState):
    return _scene_parser.run(state)

def voice_synth_node(task: dict):
    return _voice_synth.run(task)

def video_gen_node(task: dict):
    return _video_gen.run(task)

def face_swap_node(track: dict):
    return _face_swap.run(track)

def lip_sync_node(state: AgentState):
    return _lip_sync.run(state)

# Custom routers for Send() parallelism
def route_parser_to_synthesis(state: AgentState):
    """
    Branch out to multiple video and audio synthesizer nodes concurrently
    using the Send API.
    """
    tasks = state.get("scene_tasks", [])
    sends = []
    for t in tasks:
        sends.append(Send("voice_synth_node", t))
        sends.append(Send("video_gen_node", t))
    return sends

def route_video_to_face_swap(state: AgentState):
    """
    Fan-out the generated base videos into the face swapping node.
    """
    tracks = state.get("video_tracks", [])
    sends = [Send("face_swap_node", t) for t in tracks]
    return sends

def build_workflow():
    graph = StateGraph(AgentState)
    
    # Nodes
    graph.add_node("scene_parser_node", scene_parser_node)
    graph.add_node("voice_synth_node", voice_synth_node)
    graph.add_node("video_gen_node", video_gen_node)
    graph.add_node("face_swap_node", face_swap_node)
    graph.add_node("lip_sync_node", lip_sync_node)
    
    graph.set_entry_point("scene_parser_node")
    
    # scene_parser_node --(parallel)--> voice/video nodes
    graph.add_conditional_edges("scene_parser_node", route_parser_to_synthesis, ["voice_synth_node", "video_gen_node"])
    
    # Voice synth finishes, it just waits for lip sync. We don't link it directly with Send, it merges back into State automatically.
    # We must explicitly map video_gen_node to face_swap.
    # Since video_gen_node returns dict that reduces into state, we can add a conditional edge from video_gen_node or a dummy sync node.
    
    graph.add_node("sync_video_node", lambda state: state) # Dummy to wait for video
    graph.add_edge("video_gen_node", "sync_video_node")
    graph.add_conditional_edges("sync_video_node", route_video_to_face_swap, ["face_swap_node"])
    
    # Fan in: wait for face swap & voice to finish, then lip sync
    graph.add_node("pre_lip_sync_node", lambda state: state)
    graph.add_edge("voice_synth_node", "pre_lip_sync_node")
    graph.add_edge("face_swap_node", "pre_lip_sync_node")
    
    graph.add_edge("pre_lip_sync_node", "lip_sync_node")
    graph.add_edge("lip_sync_node", END)
    
    return graph.compile()

workflow = build_workflow()
