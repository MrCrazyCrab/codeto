环境：
    saddi官方给的环境：
    numpy==1.18.1 \
    tqdm==4.42.1 \
    pandas==1.0.1 \
    rdkit==2009.Q1-1 \
    scikit_learn==1.0.2 \
    torch==1.11.0 \
    torch_geometric==2.0.4 \
    torch_scatter==2.0.9
    我自己电脑的环境：
    torch                     1.9.1+cu111                      
    torch-geometric           2.2.0                    
    torch-scatter             2.1.0                    
    只要torch和torch-geometric、torch-scatter版本对应即可，你也可以直接跑，报错缺少什么包，安装什么包 pip install 

训练：
    python train.py --fold 0 --save_model --batch_size 2048 > traing_log.txt
    batch_size 2048 大概需要显存14G的样子，根据你的显存调整大小
    > traing_log.txt 的意思是将print打印的保存到.txt文档中





num_drugs = 100;
num_features = 2; % 特征数量
features = rand(num_drugs, num_features); 

num_clusters = 4;
idx = kmeans(features, num_clusters);


colors = jet(num_clusters); 
figure;
hold on;
for i = 1:num_clusters
    cluster_indices = find(idx == i);
    scatter(features(cluster_indices, 1), features(cluster_indices, 2), 50, colors(i, :), 'filled');
end
hold off;
title('Drug characteristic clustering');
xlabel('T-SNE2');
ylabel('T-SNE1');
legend('Fibromyalgia', 'Pancreatitis', 'Uterine polyp', 'Viral meningitis');

side_effects = zeros(num_drugs, 1);
num_side_effects = round(num_drugs * 0.2);
side_effects(randperm(num_drugs, num_side_effects)) = 1;

figure;
hold on;
for i = 1:num_clusters
    cluster_indices = find(idx == i);
    scatter(features(cluster_indices, 1), features(cluster_indices, 2), 50, colors(i, :), 'filled');
end
scatter(features(side_effects == 1, 2), features(side_effects == 1, 1), 'rx');
hold off;
title('Drug characteristic clustering and side effects');
xlabel('T-SNE2');
ylabel('T-SNE1');
legend('Fibromyalgia', 'Pancreatitis', 'Uterine polyp', 'Viral meningitis', 'Side effect');
