suppressPackageStartupMessages({library("maEndToEnd")})



raw_data_dir = "palmieri_data"
anno_AE <- getAE("E-MTAB-2967", path = raw_data_dir, type = "raw")

sdrf_location <- file.path(raw_data_dir, "E-MTAB-2967.sdrf.txt")
SDRF <- read.delim(sdrf_location)

rownames(SDRF) <- SDRF$Array.Data.File
SDRF <- AnnotatedDataFrame(SDRF)


raw_data <- oligo::read.celfiles(filenames = file.path(raw_data_dir,
                                                       SDRF$Array.Data.File),
                                 verbose = FALSE, phenoData = SDRF)
stopifnot(validObject(raw_data))


Biobase::pData(raw_data) <- Biobase::pData(raw_data)[, c("Source.Name",
                                                         "Characteristics.individual.",
                                                         "Factor.Value.disease.",
                                                         "Factor.Value.phenotype.")]




palmieri_eset_norm <- oligo::rma(raw_data, target = "core")

palmieri_medians <- rowMedians(Biobase::exprs(palmieri_eset_norm))

# 2826_II, 3262_II, 3271_I, 2978_II and 3332_II

no_of_samples <-
  table(paste0(pData(palmieri_eset_norm)$Factor.Value.disease., "_",
               pData(palmieri_eset_norm)$Factor.Value.phenotype.))
no_of_samples

samples_cutoff <- min(no_of_samples)

man_threshold <- 4

idx_man_threshold <- apply(Biobase::exprs(palmieri_eset_norm), 1,
                           function(x){
                             sum(x > man_threshold) >= samples_cutoff})
table(idx_man_threshold)

palmieri_manfiltered <- subset(palmieri_eset_norm, idx_man_threshold)

anno_palmieri <- AnnotationDbi::select(hugene10sttranscriptcluster.db,
                                       keys = (featureNames(palmieri_manfiltered)),
                                       columns = c("SYMBOL", "GENENAME"),
                                       keytype = "PROBEID")


anno_palmieri <- subset(anno_palmieri, !is.na(SYMBOL))

anno_grouped <- group_by(anno_palmieri, PROBEID)
anno_summarized <-
  dplyr::summarize(anno_grouped, no_of_matches = n_distinct(SYMBOL))

anno_filtered <- filter(anno_summarized, no_of_matches == 1)

probe_stats <- anno_filtered
nrow(probe_stats)

ids_to_keep <- (featureNames(palmieri_manfiltered) %in% probe_stats$PROBEID)
table(ids_to_keep)

palmieri_final <- subset(palmieri_manfiltered, ids_to_keep)

validObject(palmieri_final)

fData(palmieri_final)$PROBEID <- rownames(fData(palmieri_final))

fData(palmieri_final) <- left_join(fData(palmieri_final), anno_palmieri)

rownames(fData(palmieri_final)) <- fData(palmieri_final)$PROBEID
validObject(palmieri_final)

individual <-
  as.character(Biobase::pData(palmieri_final)$Characteristics.individual.)

tissue <- str_replace_all(Biobase::pData(palmieri_final)$Factor.Value.phenotype.,
                          " ", "_")

tissue <- ifelse(tissue == "non-inflamed_colonic_mucosa",
                 "nI", "I")

disease <- 
  str_replace_all(Biobase::pData(palmieri_final)$Factor.Value.disease.,
                  " ", "_")

disease <- 
  ifelse(str_detect(Biobase::pData(palmieri_final)$Factor.Value.disease.,
                    "Crohn"), "CD", "UC")




outliers_klaus_extended <- c(2826, 3262, 3271, 2978,  3332)
outlier_klaus <- c(2826, 3262, 3332)

pData_palmieri <- pData(palmieri_final)
idx_filter_CD <- (! pData_palmieri$Characteristics.individual %in% outlier_klaus) &
  (disease == "CD")

#idx_filter_CD <- (disease == "CD")
i_CD <- individual[idx_filter_CD]
design_palmieri_CD <- model.matrix(~ 0 + tissue[idx_filter_CD] + i_CD)
colnames(design_palmieri_CD)[1:2] <- c("I", "nI")
rownames(design_palmieri_CD) <- i_CD

contrast_matrix_CD <- makeContrasts(I-nI, levels = design_palmieri_CD)


palmieri_fit_CD <- eBayes(contrasts.fit(lmFit(palmieri_final[,idx_filter_CD],
                                              design = design_palmieri_CD),
                                        contrast_matrix_CD))



                                                                                                     

                                                                                                     


table_CD <- topTable(palmieri_fit_CD, number = Inf)
nrow(subset(table_CD, P.Value < 0.001))
hist(table_CD$P.Value)




expr_data_CD <- Biobase::exprs(palmieri_final[, idx_filter_CD])
tissue_CD <- tissue[idx_filter_CD]
individual_CD <- individual[idx_filter_CD]

I_individuals <- individual_CD[tissue_CD == "I"]
nI_individuals <- individual_CD[tissue_CD == "nI"]

cat("Are they in same order?", identical(I_individuals, nI_individuals), "\n")


I_samples <- expr_data_CD[, tissue_CD == "I"]
nI_samples <- expr_data_CD[, tissue_CD == "nI"]

I_individuals_ordered <- I_individuals[order(I_individuals)]
nI_individuals_ordered <- nI_individuals[order(nI_individuals)]

cat("After ordering - same individuals?", identical(I_individuals_ordered, nI_individuals_ordered), "\n")

diff_matrix <- I_samples - nI_samples
colnames(diff_matrix) <- individual_CD[tissue_CD == "I"]

fData_palmieri <-  fData(palmieri_final)
ttest_results <- rowttests(diff_matrix)

full_tbl <- cbind(fData_palmieri, diff_matrix)
library(readr)
write_csv(full_tbl, "palmieri_pairwise.csv")
hist(ttest_results$p.value)
sum(table_CD$P.Value < 0.001)
sum(ttest_results$p.value < 0.001)

fData_palmieri_sig <-fData_palmieri[ttest_results$p.value < 0.001,]

bla<- cbind(fData_palmieri_sig, ttest_results[ttest_results$p.value < 0.001,])
