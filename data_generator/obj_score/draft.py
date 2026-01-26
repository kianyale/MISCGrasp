import trimesh

# 加载3D模型
# mesh = trimesh.load('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/data/urdfs/egad_test/A0.obj')  # 替换为你的模型文件路径
# AABB Extents: [0.09227328 0.08459109 0.10007293]
mesh = trimesh.load('/media/yons/6d379145-c1d8-430f-9056-7777219c83a8/MISCGrasp/data/assets/data/urdfs/pile/test/soccer_ball_poisson_000_visual.obj')  # 替换为你的模型文件路径
# AABB Extents: [0.07       0.07044701 0.06841892]
# 计算轴对齐包围盒 (AABB)
aabb = mesh.bounding_box
aabb_extents = aabb.extents  # 包围盒的尺寸
aabb_corners = aabb.vertices  # AABB的8个顶点

print("AABB Extents:", aabb_extents)
print("AABB Corners:\n", aabb_corners)

# 计算最小体积包围盒 (OBB)
obb = mesh.bounding_box_oriented
obb_extents = obb.extents  # 包围盒的尺寸
obb_corners = obb.vertices  # OBB的8个顶点
obb_transform = obb.primitive.transform  # OBB的变换矩阵

print("OBB Extents:", obb_extents)
print("OBB Corners:\n", obb_corners)
print("OBB Transform Matrix:\n", obb_transform)

# 可视化包围盒
scene = trimesh.Scene([mesh, aabb.to_mesh(), obb.to_mesh()])
scene.show()

