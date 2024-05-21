library(SpaTalk)
options(warn = 0)  
# load starmap data
args = commandArgs(T)
st_dir <- args[1]
st_meta_dir <- args[2]
sc_coord_dir <- args[3]
meta_key <- args[4]
species <- args[5]
out_f <- args[6]
out_f = paste0(out_f,"/")
print(args)

if (length(args) > 6){
    n_cores = strtoi(args[7])
}else{
    n_cores = 4
}
# print(n_cores)

# TODO
args <- commandArgs(trailingOnly = FALSE)
scriptPath <- normalizePath(sub("^--file=", "", args[grep("^--file=", args)]))
scriptPath <- dirname(scriptPath)
##########
dir.create(file.path(out_f), showWarnings = FALSE)
if (file.exists(out_f)) {
  print("The file exists")
} else {
  print("The file does not exist")
}
print(out_f)
# print(meta_key)
if (grepl('csv', st_dir)){
    st_data = t(read.table(file = st_dir, sep = ',', header = TRUE,row.names = 1))
} else{
    st_data = t(read.table(file = st_dir, sep = '\t', header = TRUE,row.names = 1))
}

if (grepl('csv', st_meta_dir)){
    st_meta = read.table(file = st_meta_dir, sep = ',', header = TRUE,row.names = 1)
} else{
    st_meta = read.table(file = st_meta_dir, sep = '\t', header = TRUE,row.names = 1)
}

if (grepl('csv', sc_coord_dir)){
    sc_coord = read.table(file = sc_coord_dir, sep = ',', header = TRUE,row.names = 1)
} else{
    sc_coord = read.table(file = sc_coord_dir, sep = '\t', header = TRUE,row.names = 1)
}


if (species == 'Mouse'){
    if (grepl('spex', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else if (grepl('before', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else{
        sc_coord = sc_coord[c('x','y')]
    }
}

if (species == 'Human'){
    if (grepl('spex', sc_coord_dir)){
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }else{
        sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
    }
}

# if (grepl('mela', sc_coord_dir)){
#     if (grepl('spex', sc_coord_dir)){
#         sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
#     }else{
#         sc_coord = sc_coord[c('adj_spex_UMAP1','adj_spex_UMAP2')]
#     }
# }
print('loaded')

if (grepl('Human', species)){
    max_hop = 3
}else if ( 
    grepl('Mouse', species)){
    max_hop = 4
    # add neuronchat db
    lr_df_dir = paste0(scriptPath,"/../LR/mouse_LR_pairs.txt")
    new_lr_df = read.table(file = lr_df_dir, sep = '\t', header = FALSE)
    new_lr_df = new_lr_df[,c('V1','V2')]
    colnames(new_lr_df) = c('ligand','receptor')
    species = 'Mouse'
    new_lr_df$species = species
    new_lrpairs = rbind(lrpairs,new_lr_df)
    new_lrpairs$ligand = gsub("_", "-", new_lrpairs$ligand)
    new_lrpairs$receptor = gsub("_", "-", new_lrpairs$receptor)
    new_lrpairs = unique(new_lrpairs)
    lrpairs = new_lrpairs
}



# subset by meta index
st_data = st_data[,rownames(sc_coord)]
# Formating
sc_coord$cell = rownames(sc_coord)
sc_coord$cell <- sub("^", "C",sc_coord$cell)
colnames(sc_coord) = c('x','y','cell')
sc_coord = sc_coord[,c('cell','x','y')]

colnames(st_data) = sc_coord$cell
colnames(st_data) = gsub("_", "-", colnames(st_data))
rownames(st_data) = gsub("_", "-", rownames(st_data))
st_data = as.data.frame(st_data)

obj <- createSpaTalk(st_data = as.matrix(st_data),
                     st_meta = sc_coord,
                     species = species,
                     if_st_is_sc = T,
                     spot_max_cell = 1,celltype = st_meta[[meta_key]])
tp_lst = unique(obj@meta$rawmeta$celltype)


obj <- find_lr_path(object = obj , lrpairs = lrpairs, pathways = pathways, if_doParallel = T, use_n_cores=n_cores, max_hop = max_hop)

for (tp1 in tp_lst) {
  for (tp2 in tp_lst) {
    if (tp1 != tp2) {
      tryCatch({
        obj <- dec_cci(object = obj, celltype_sender = tp1, celltype_receiver = tp2,
                       if_doParallel = T, use_n_cores = n_cores, pvalue = 0.1, n_neighbor = 20,
                       co_exp_ratio = 0.05, min_pairs = 2)
        obj <- dec_cci(object = obj, celltype_sender = tp2, celltype_receiver = tp1,
                       if_doParallel = T, use_n_cores = n_cores, pvalue = 0.1, n_neighbor = 20,
                       co_exp_ratio = 0.05, min_pairs = 2)
        print(tp1)
        print(tp2)
      }, error = function(e) {
        cat("Error occurred during iteration: tp1:", tp1, "tp2:", tp2, "Error:", conditionMessage(e), "\n")
      })
      next
    }
  }
}

# obj <- dec_cci_all(object = obj, if_doParallel = T, use_n_cores=n_cores, pvalue=0.1, n_neighbor = 20, co_exp_ratio=0.05,min_pairs=2)
write.csv(obj@lrpair, paste0(out_f,"/lr_pair.csv"), row.names = TRUE,quote = F)
saveRDS(obj, paste0(out_f,"/spatalk.rds"))
############## LR ana ###################
# obj = readRDS('spatalk.rds')
# out_f = './'
r_object = obj@cellpair
df <- data.frame(
  Name = character(),
  cell_sender = character(),
  cell_receiver = character(),
  stringsAsFactors = FALSE
)

for (name in names(r_object)) {
  sender <- r_object[[name]]$cell_sender
  receiver <- r_object[[name]]$cell_receiver
  df <- rbind(df, data.frame(Name = rep(name, length(sender)), cell_sender = sender, cell_receiver = receiver, stringsAsFactors = FALSE))
}
write.csv(df, paste0(out_f,"/cellpair.csv"), row.names = T,quote = F)
write.csv(obj@meta$rawmeta, paste0(out_f,"/spatalk_meta.csv"), row.names = T,quote = F)