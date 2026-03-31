import pickle

stub_path = 'stubs/track_stubs.pkl'
with open(stub_path, 'rb') as f:
    tracks = pickle.load(f)

print('=== TRACK STUBS CONTENTS ===')
for key, val in tracks.items():
    n_frames = len(val)
    ids = set()
    for fd in val:
        ids.update(fd.keys())
    id_sample = sorted(ids)[:10]
    print(f'  {key}: {n_frames} frames, IDs (sample)={id_sample}')
    for fd in val:
        if fd:
            first_id = next(iter(fd))
            print(f'    sample field keys: {list(fd[first_id].keys())}')
            break

print()
total_frames = len(tracks.get('players', []))
print(f'Total player frames: {total_frames}')
