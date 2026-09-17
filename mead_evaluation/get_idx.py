import numpy as np

idx_all = '/data/fdm/region/all_face.txt'
idx_all = open(idx_all, 'r')
idx_all = idx_all.readlines()
idx_all = [idx.strip("\n") for idx in idx_all]

idx_lip = '/data/fdm/region/lip.txt'
idx_lip = open(idx_lip, 'r')
idx_lip = idx_lip.readlines()
idx_lip = [idx.strip("\n") for idx in idx_lip]

all_id = []
for idx in idx_all:
    id = idx[:-1].split(',')

    for i in id:
        all_id.append(int(i.strip(' ')))

all_lip_id = []
for idx in idx_lip:
    id = idx[:-1].split(',')

    for i in id:
        all_lip_id.append(int(i.strip(' ')))

all_id = list(set(all_id)) # 去除重复的id
all_id = np.array(all_id)

all_lip_id = list(set(all_lip_id)) # 去除重复的id
all_lip_id = np.array(all_lip_id)

all_emotion_id = np.setdiff1d(all_id, all_lip_id) # 求差集

print(all_id.shape)
print(all_lip_id.shape)
print(all_emotion_id.shape)

# np.save('/data/WX/fdm/region/face_vertices.npy', all_id)
# np.save('/data/WX/fdm/region/lip_vertices.npy', all_lip_id)
np.save('/data/WX/fdm/region/emotion_vertices.npy', all_emotion_id)