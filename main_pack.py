import sys
sys.path.append('./PFRTool')
import PFRTool as pt

def main(config_path):
    pt.run(config_path)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]  # Get the configuration file path from the command line
    main(config_path)