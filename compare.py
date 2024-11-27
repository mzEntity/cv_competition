import os

def compare(answer_path, truth_path):
    answer_map = {}
    truth_map = {}
    with open(answer_path, 'r') as file:
        for line in file:
            img_name, label = line.strip().split()
            answer_map[img_name] = int(label)
            
    with open(truth_path, 'r') as file:
        for line in file:
            img_name, label = line.strip().split()
            truth_map[img_name] = int(label)
    
    total = len(answer_map)
    if total != len(truth_map):
        print("image_label_pair count not match")
        
    right = 0
    mae = 0
    mse = 0
    for name, label in answer_map.items():
        diff = abs(label - truth_map[name])
        if diff == 0:
            right += 1
        mse += diff ** 2
        mae += diff
    
    print(f"right_count: {right} / {total}")
    print(f"mse: {mse / total}")
    print(f"mae: {mae / total}")
    
    
if __name__ == "__main__":
    root_path = "D:/fudan/2024Autumn/CV/competition/cv_competition"
    answer_path = os.path.join(root_path, "output.txt")
    truth_path = os.path.join(root_path, "data/annotations/val.txt")
    
    compare(answer_path, truth_path)