"""
Main entry point for PROJECT MONTAGE Phase 2: The Studio Floor
Converts phase 1 JSON manifests to multimodal scenes.
"""
import argparse
import sys
import os
from config import config
from graph.state import get_initial_state
from graph.workflow import workflow

def main():
    parser = argparse.ArgumentParser(description="Phase 2 - The Studio Floor")
    parser.add_argument("--manifest", required=True, help="Path to scene_manifest.json")
    args = parser.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERROR] Manifest not found: {args.manifest}")
        sys.exit(1)

    # Clean raw_scenes from previous runs to avoid stale file pollution
    import shutil
    if os.path.exists(config.raw_scenes_dir):
        shutil.rmtree(config.raw_scenes_dir)
    os.makedirs(config.raw_scenes_dir, exist_ok=True)
    os.makedirs(config.faiss_index_path, exist_ok=True)

    print("\n" + "="*60)
    print("  === PROJECT MONTAGE - The Studio Floor (Phase 2)")
    print("="*60)
    print(f"  Manifest  : {args.manifest}")
    print("="*60 + "\n")

    initial_state = get_initial_state(manifest_path=args.manifest)
    
    try:
        final_state = workflow.invoke(initial_state)
        
        print("\n" + "="*60)
        print("  === AUDIOVISUAL GENERATION COMPLETE")
        print("="*60)
        final_vids = final_state.get("final_videos", [])
        print(f"  Generated Videos : {len(final_vids)}")
        for i, vid in enumerate(final_vids):
            print(f"  [{i+1}] {vid.get('video_file')}")
        print("="*60)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
