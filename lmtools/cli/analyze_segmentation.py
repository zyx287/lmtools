'''
CLI command to analyze segmentation masks and generate statistics.
'''
import argparse
from lmtools.seg import analyze_segmentation, summarize_segmentation, get_bounding_boxes
import json

def add_arguments(parser):
    '''
    Add command line arguments for the analyze_segmentation command.
    '''
    parser.add_argument(
        "mask_path", 
        type=str, 
        help="Path to the segmentation mask .npy file"
    )
    parser.add_argument(
        "--no-stats", 
        action="store_true", 
        help="Don't compute per-object statistics"
    )
    parser.add_argument(
        "--min-size", 
        type=int, 
        default=0, 
        help="Minimum object size to include (pixels)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Save results to this JSON file"
    )
    parser.add_argument(
        "--bbox", 
        action="store_true", 
        help="Get bounding boxes for objects instead of full analysis"
    )
    parser.add_argument(
        "--label", 
        type=int, 
        help="Specific label to get bounding box for (requires --bbox)"
    )

def main(args):
    '''
    Execute the analyze_segmentation command.
    '''
    try:
        if args.bbox:
            # Get bounding boxes
            bboxes = get_bounding_boxes(
                args.mask_path,
                label_id=args.label,
                return_format='coords'
            )
            
            if args.label:
                print(f"Bounding box for label {args.label}: {bboxes}")
            else:
                print(f"Found {len(bboxes)} bounding boxes")
                if len(bboxes) > 0:
                    print("First 5 bounding boxes:")
                    for i, (label, bbox) in enumerate(list(bboxes.items())[:5]):
                        print(f"  Label {label}: {bbox}")
            
            # Save to JSON if requested
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(bboxes, f, indent=2)
                print(f"Bounding boxes saved to {args.output}")
        else:
            # Run standard segmentation analysis
            results = analyze_segmentation(
                args.mask_path, 
                compute_object_stats=not args.no_stats,
                min_size=args.min_size
            )
            
            summarize_segmentation(results)
            
            if args.output:
                # Convert numpy types to Python native types for JSON serialization
                def convert_for_json(obj):
                    if isinstance(obj, (list, tuple)) and len(obj) > 0 and hasattr(obj[0], 'tolist'):
                        return [item.tolist() if hasattr(item, 'tolist') else item for item in obj]
                    elif hasattr(obj, 'tolist'):
                        return obj.tolist()
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    elif hasattr(obj, 'to_dict'):
                        return obj.to_dict(orient='records')
                    return obj
                
                # Convert results to JSON-serializable format
                json_results = {k: convert_for_json(v) for k, v in results.items()}
                
                with open(args.output, 'w') as f:
                    json.dump(json_results, f, indent=2)
                    
                print(f"Results saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        raise