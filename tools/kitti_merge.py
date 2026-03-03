import os
import argparse

def merge_trajectories(input_path1, input_path2, output_path):
    assert sorted(os.listdir(input_path1)) == sorted(os.listdir(input_path2))

    filenames = sorted(os.listdir(input_path1))
    filenames = [f for f in filenames if f.endswith('.txt')]
    for filename in filenames:
        print(f"Merging {filename} from {input_path1} and {input_path2} into {output_path}")
        file1_path = os.path.join(input_path1, filename)
        file2_path = os.path.join(input_path2, filename)

        with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
        merged_lines = lines1 + lines2

        output_file_path = os.path.join(output_path, filename)
        os.makedirs(output_path, exist_ok=True)
        with open(output_file_path, 'w') as out_file:
            out_file.write(''.join(merged_lines))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Merge trajectory files.')
    parser.add_argument('--input_path1', type=str,
                        default='output/tracking_results/fusion_pointgnn_Car_test',
                        help='Directory containing trajectory files to merge')
    parser.add_argument('--input_path2', type=str,
                        default='output/tracking_results/fusion_pointgnn_Pedestrian_test',
                        help='Directory containing trajectory files to merge')
    parser.add_argument('--output_path', type=str,
                        default='output/tracking_results/fusion_pointgnn_test',
                        help='Output directory for merged trajectories')

    args = parser.parse_args()

    merge_trajectories(args.input_path1, args.input_path2, args.output_path)