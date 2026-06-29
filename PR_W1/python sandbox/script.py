import re
import matplotlib.pyplot as plt

def plot_from_report(file_path):
    x_coords = []
    y_coords = []
    
    # 1. Read and deconstruct the input file
    with open(file_path, 'r') as f:
        for line in f:
            # Matches optional spaces, followed by #<num>:, then extracts the numbers inside {x, y}
            match = re.search(r'#\d+:\s*\{([^,]+),\s*([^}]+)\}', line)
            if match:
                x_val = float(match.group(1).strip())
                y_val = float(match.group(2).strip())
                x_coords.append(x_val)
                y_coords.append(y_val)
                
    if not x_coords:
        print(f"No valid data points found in {file_path}. Please check the file format.")
        return

    # 2. Plot the graph like specified
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot the parsed points
    ax.scatter(x_coords, y_coords, color='blue', edgecolors='black', zorder=3)
    
    # Grid limits as requested
    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-4.5, 1.5)
    
    # Style gridlines (no title or point names)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', zorder=0)
    
    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    plt.tight_layout()
    
    # Save the output image
    output_filename = 'differential_evolution_plot.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully processed {len(x_coords)} points and saved plot to '{output_filename}'.")

# To run the code, ensure 'report.txt' is in the same directory and execute:
if __name__ == "__main__":
    plot_from_report('report.txt')