import os
import json
import argparse

def get_fixes_path(video):
    return os.path.join("data/annotations/remappings", f"{video}_fixes.json")

def load_fixes(video):
    path = get_fixes_path(video)
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def save_fixes(video, fixes):
    path = get_fixes_path(video)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(fixes, f, indent=4)

def add_fix(video, clip, start, end, remap_dict):
    fixes = load_fixes(video)
    
    if clip not in fixes:
        fixes[clip] = []
    
    # Avoid duplicate ranges
    fixes[clip] = [r for r in fixes[clip] if not (r['frame_start'] == start and r['frame_end'] == end)]
    
    # Add the new ranged remap
    fixes[clip].append({
        "frame_start": start,
        "frame_end": end,
        "remap": remap_dict
    })
    
    # Sort ranges by frame_start
    fixes[clip].sort(key=lambda x: x['frame_start'])
    
    save_fixes(video, fixes)
    print(f"✓ Added manual fix to {get_fixes_path(video)} for clip {clip} [{start}-{end}]")

def list_fixes(video):
    fixes = load_fixes(video)
    if not fixes:
        print(f"No manual fixes found for video {video}.")
        return
    
    print(f"\nManual fixes for video: {video}")
    for clip, steps in fixes.items():
        print(f"  Clip {clip}:")
        if isinstance(steps, list):
            for i, s in enumerate(steps):
                start = s.get('frame_start', 'All')
                end = s.get('frame_end', 'All')
                remap = s.get('remap', 'N/A')
                delete = s.get('delete', 'N/A')
                print(f"    Step {i} [{start}-{end}]: Remap: {remap}, Delete: {delete}")
        else:
            print(f"    Warning: Clip {clip} is not in list format. Content: {steps}")

def clear_fixes(video):
    path = get_fixes_path(video)
    if os.path.exists(path):
        os.remove(path)
        print(f"✓ Deleted fix file for video {video}: {path}")
    else:
        print(f"No fix file found for {video}.")

def main():
    parser = argparse.ArgumentParser(description="Manage manual ID overrides per video.")
    subparsers = parser.add_subparsers(dest="command")
    
    add_p = subparsers.add_parser("add")
    add_p.add_argument("--video", required=True)
    add_p.add_argument("--clip", required=True)
    add_p.add_argument("--start", type=int, required=True)
    add_p.add_argument("--end", type=int, required=True)
    add_p.add_argument("--remap", required=True, help="JSON string of the remap, e.g. '{\"0\": \"3\"}'")
    
    list_p = subparsers.add_parser("list")
    list_p.add_argument("--video", required=True)
    
    clear_p = subparsers.add_parser("clear")
    clear_p.add_argument("--video", required=True)
    
    args = parser.parse_args()
    
    if args.command == "add":
        try:
            remap_dict = json.loads(args.remap)
            add_fix(args.video, args.clip, args.start, args.end, remap_dict)
        except json.JSONDecodeError:
            print("Error: The --remap argument must be a valid JSON string.")
    elif args.command == "list":
        list_fixes(args.video)
    elif args.command == "clear":
        clear_fixes(args.video)

if __name__ == "__main__":
    main()
