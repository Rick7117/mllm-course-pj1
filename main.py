import argparse

def main():
    parser = argparse.ArgumentParser(description='Cross-Modal Alignment Analysis Project')
    parser.add_argument('--task', type=str, choices=['retrieval', 'captioning', 'representation', 'nearest_neighbor', 'all'],
                        default='all', help='Task to run')
    
    args = parser.parse_args()
    
    if args.task == 'retrieval' or args.task == 'all':
        from task_retrieval import run_retrieval
        print("Running Retrieval Task...")
        run_retrieval()
    
    if args.task == 'captioning' or args.task == 'all':
        from task_captioning import run_captioning
        print("\nRunning Captioning Task...")
        run_captioning()
    
    if args.task == 'representation' or args.task == 'all':
        from task_representation import run_visualization
        print("\nRunning Representation Analysis Task...")
        run_visualization()
    
    if args.task == 'nearest_neighbor' or args.task == 'all':
        from task_nearest_neighbor import run_nearest_neighbor_analysis, analyze_compositional_generalization
        print("\nRunning Nearest Neighbor Analysis...")
        run_nearest_neighbor_analysis()
        print("\nRunning Compositional Generalization Analysis...")
        analyze_compositional_generalization()
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()