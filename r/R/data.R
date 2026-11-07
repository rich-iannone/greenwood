# The bundled datasets, written by scripts/export_r_datasets.R. tests/testthat/test-data.R checks
# their shapes and codings.

#' NCCTG lung cancer data
#'
#' Survival of 228 patients with advanced lung cancer in the North Central Cancer Treatment
#' Group. The performance scores rate how well a patient can carry out ordinary daily activities.
#'
#' @format A data frame with 228 rows and 10 columns:
#' \describe{
#'   \item{inst}{Institution code.}
#'   \item{time}{Survival time in days.}
#'   \item{status}{Censoring status: 1 is censored, 2 is dead.}
#'   \item{age}{Age in years.}
#'   \item{sex}{1 is male, 2 is female.}
#'   \item{ph.ecog}{ECOG performance score rated by the physician, from 0 (asymptomatic) to 4
#'     (bedbound).}
#'   \item{ph.karno}{Karnofsky performance score rated by the physician, from 0 (bad) to 100
#'     (good).}
#'   \item{pat.karno}{Karnofsky performance score rated by the patient.}
#'   \item{meal.cal}{Calories consumed at meals.}
#'   \item{wt.loss}{Weight loss in the last six months, in pounds.}
#' }
#'
#' The 1/2 coding of `status` reads as an event indicator directly, so `Surv(time, status)` is the
#' response.
#'
#' @source Loprinzi CL, Laurie JA, Wieand HS, et al. (1994). Prospective evaluation of prognostic
#'   variables from patient-completed questionnaires. *Journal of Clinical Oncology*, 12(3),
#'   601-607.
#'
#' @examples
#' kaplan_meier(Surv(time, status) ~ sex, data = lung)
"lung"

#' Veterans' Administration lung cancer trial
#'
#' A randomized trial of two treatment regimens for lung cancer, in 137 patients.
#'
#' @format A data frame with 137 rows and 8 columns:
#' \describe{
#'   \item{trt}{Treatment: 1 is standard, 2 is test.}
#'   \item{celltype}{Cell type, a factor with levels `squamous`, `smallcell`, `adeno`, and
#'     `large`.}
#'   \item{time}{Survival time in days.}
#'   \item{status}{Censoring status: 0 is censored, 1 is dead.}
#'   \item{karno}{Karnofsky performance score (100 is good).}
#'   \item{diagtime}{Months from diagnosis to randomization.}
#'   \item{age}{Age in years.}
#'   \item{prior}{Prior therapy: 0 is no, 10 is yes.}
#' }
#'
#' @source Kalbfleisch JD and Prentice RL (1980). *The Statistical Analysis of Failure Time Data*.
#'   Wiley, New York.
#'
#' @examples
#' cox_ph(Surv(time, status) ~ trt + celltype + karno, data = veteran)
"veteran"

#' Ovarian cancer survival
#'
#' Survival in a randomized trial comparing two treatments for ovarian cancer, in 26 patients.
#'
#' @format A data frame with 26 rows and 6 columns:
#' \describe{
#'   \item{futime}{Survival or censoring time in days.}
#'   \item{fustat}{Censoring status: 0 is censored, 1 is dead.}
#'   \item{age}{Age in years.}
#'   \item{resid.ds}{Residual disease present: 1 is no, 2 is yes.}
#'   \item{rx}{Treatment group, 1 or 2.}
#'   \item{ecog.ps}{ECOG performance status (1 is better).}
#' }
#'
#' @source Edmunson JH, Fleming TR, Decker DG, et al. (1979). Different chemotherapeutic
#'   sensitivities and host factors affecting prognosis in advanced ovarian carcinoma versus
#'   minimal residual disease. *Cancer Treatment Reports*, 63, 241-247.
#'
#' @examples
#' logrank_test(Surv(futime, fustat) ~ rx, data = ovarian)
"ovarian"

#' Mayo Clinic primary biliary cholangitis data
#'
#' The Mayo Clinic trial in primary biliary cholangitis (PBC), conducted between 1974 and 1984.
#' The first 312 patients took part in the randomized placebo-controlled trial of D-penicillamine.
#' The other 106 didn't, but consented to have basic measurements recorded and to be followed for
#' survival. [pbcseq] holds the follow-up laboratory values of the trial patients.
#'
#' @format A data frame with 418 rows and 20 columns:
#' \describe{
#'   \item{id}{Case number.}
#'   \item{time}{Days from registration to the earlier of death, transplantation, or the study
#'     analysis in July 1986.}
#'   \item{status}{Status at the end point: 0 is censored, 1 is transplant, 2 is dead.}
#'   \item{trt}{Treatment: 1 is D-penicillamine, 2 is placebo, `NA` is not randomized.}
#'   \item{age}{Age in years.}
#'   \item{sex}{A factor with levels `m` and `f`.}
#'   \item{ascites}{Presence of ascites.}
#'   \item{hepato}{Presence of hepatomegaly (an enlarged liver).}
#'   \item{spiders}{Blood vessel malformations in the skin.}
#'   \item{edema}{0 is no edema, 0.5 is untreated or successfully treated, 1 is edema despite
#'     diuretic therapy.}
#'   \item{bili}{Serum bilirubin (mg/dl).}
#'   \item{chol}{Serum cholesterol (mg/dl).}
#'   \item{albumin}{Serum albumin (g/dl).}
#'   \item{copper}{Urine copper (ug/day).}
#'   \item{alk.phos}{Alkaline phosphatase (U/liter).}
#'   \item{ast}{Aspartate aminotransferase (U/ml).}
#'   \item{trig}{Triglycerides (mg/dl).}
#'   \item{platelet}{Platelet count.}
#'   \item{protime}{Standardized blood clotting time.}
#'   \item{stage}{Histologic stage of disease.}
#' }
#'
#' Death is `Surv(time, status == 2)`, with a transplant counted as censoring. For transplant and
#' death as competing events, use `Surv(time, factor(status, 0:2, c("censor", "transplant",
#' "death")))`.
#'
#' @source Therneau T and Grambsch P (2000). *Modeling Survival Data: Extending the Cox Model*.
#'   Springer, New York.
#'
#' @examples
#' cox_ph(Surv(time, status == 2) ~ age + edema + log(bili) + albumin, data = pbc)
"pbc"

#' Mayo Clinic primary biliary cholangitis, sequential laboratory data
#'
#' The follow-up laboratory values of the 312 randomized patients of [pbc], one row per visit.
#' Lab values were taken at protocol visits (six months, one year, and yearly after that) and at
#' extra visits, which often came with a worsening condition and didn't always include every test.
#' So the missing values near the end of follow-up may be informative.
#'
#' @format A data frame with 1945 rows and 19 columns:
#' \describe{
#'   \item{id}{Case number, as in [pbc].}
#'   \item{futime}{Days from registration to death, transplantation, or the end of the study.}
#'   \item{status}{Status at the end point: 0 is censored, 1 is transplant, 2 is dead.}
#'   \item{trt}{Treatment: 1 is D-penicillamine, 2 is placebo.}
#'   \item{age}{Age in years.}
#'   \item{sex}{A factor with levels `m` and `f`.}
#'   \item{day}{Days from enrollment to this visit. The measurements below are from this visit.}
#'   \item{ascites, hepato, spiders, edema, bili, chol, albumin, alk.phos, ast, platelet,
#'     protime, stage}{The laboratory and clinical values, as in [pbc].}
#' }
#'
#' [split_episodes()] builds the counting-process data for a model with time-varying covariates.
#'
#' @source Murtaugh PA, Dickson ER, Van Dam GM, et al. (1994). Primary biliary cirrhosis:
#'   prediction of short-term survival based on repeated patient visits. *Hepatology*, 20(1),
#'   126-134.
#'
#' @examples
#' head(pbcseq[c("id", "futime", "status", "day", "bili")])
"pbcseq"

#' Chemotherapy for stage B/C colon cancer
#'
#' One of the first successful trials of adjuvant chemotherapy for colon cancer, comparing
#' observation, levamisole, and levamisole with 5-FU. There are two rows per patient, one for
#' recurrence and one for death.
#'
#' @format A data frame with 1858 rows and 16 columns:
#' \describe{
#'   \item{id}{Patient id.}
#'   \item{study}{1 for every patient.}
#'   \item{rx}{Treatment, a factor with levels `Obs` (observation), `Lev` (levamisole), and
#'     `Lev+5FU`.}
#'   \item{sex}{1 is male.}
#'   \item{age}{Age in years.}
#'   \item{obstruct}{Obstruction of the colon by the tumour.}
#'   \item{perfor}{Perforation of the colon.}
#'   \item{adhere}{Adherence to nearby organs.}
#'   \item{nodes}{Number of lymph nodes with detectable cancer.}
#'   \item{status}{Censoring status: 0 is censored, 1 is the event.}
#'   \item{differ}{Differentiation of the tumour: 1 is well, 2 is moderate, 3 is poor.}
#'   \item{extent}{Extent of local spread: 1 is submucosa, 2 is muscle, 3 is serosa, 4 is
#'     contiguous structures.}
#'   \item{surg}{Time from surgery to registration: 0 is short, 1 is long.}
#'   \item{node4}{More than four positive lymph nodes.}
#'   \item{time}{Days until the event or censoring.}
#'   \item{etype}{Event type: 1 is recurrence, 2 is death.}
#' }
#'
#' @source Moertel CG, Fleming TR, Macdonald JS, et al. (1990). Levamisole and fluorouracil for
#'   adjuvant therapy of resected colon carcinoma. *New England Journal of Medicine*, 322,
#'   352-358.
#'
#' @examples
#' deaths <- colon[colon$etype == 2, ]
#' kaplan_meier(Surv(time, status) ~ rx, data = deaths)
"colon"

#' Monoclonal gammopathy data
#'
#' The natural history of 1384 patients with monoclonal gammopathy of undetermined significance
#' (MGUS), followed for progression to a plasma-cell malignancy (PCM) and for death. It's a
#' competing-risks dataset, since death before progression precludes it.
#'
#' @format A data frame with 1384 rows and 11 columns:
#' \describe{
#'   \item{id}{Subject id.}
#'   \item{age}{Age at diagnosis, in years.}
#'   \item{sex}{A factor with levels `F` and `M`.}
#'   \item{dxyr}{Year of diagnosis.}
#'   \item{hgb}{Hemoglobin.}
#'   \item{creat}{Creatinine.}
#'   \item{mspike}{Size of the monoclonal serum spike.}
#'   \item{ptime}{Months until progression to PCM or last contact.}
#'   \item{pstat}{Progression to PCM: 0 is no, 1 is yes.}
#'   \item{futime}{Months until death or last contact.}
#'   \item{death}{Death: 0 is no, 1 is yes.}
#' }
#'
#' [first_event()] builds the competing-risks response from the two pairs of columns.
#'
#' @source Kyle R, Therneau T, Rajkumar V, et al. (2002). A long-term study of prognosis in
#'   monoclonal gammopathy of undetermined significance. *New England Journal of Medicine*, 346,
#'   564-569.
#'
#' @examples
#' head(mgus2)
"mgus2"

#' ECOG melanoma trial E1684
#'
#' The Eastern Cooperative Oncology Group trial E1684 of high-dose interferon alfa-2b as adjuvant
#' therapy for melanoma, in 284 patients. Many patients never relapse, which makes it a standard
#' dataset for cure models.
#'
#' @format A data frame with 284 rows and 5 columns:
#' \describe{
#'   \item{TRT}{Treatment: 0 is control, 1 is interferon.}
#'   \item{FAILTIME}{Relapse-free time in years.}
#'   \item{FAILCENS}{Relapse indicator: 0 is censored, 1 is relapse.}
#'   \item{AGE}{Age, centered at its mean.}
#'   \item{SEX}{0 is male, 1 is female.}
#' }
#'
#' @source Kirkwood JM, Strawderman MH, Ernstoff MS, et al. (1996). Interferon alfa-2b adjuvant
#'   therapy of high-risk resected cutaneous melanoma: the Eastern Cooperative Oncology Group
#'   Trial EST 1684. *Journal of Clinical Oncology*, 14(1), 7-17.
#'
#' @examples
#' mixture_cure(Surv(FAILTIME, FAILCENS) ~ TRT, cure = ~TRT, data = e1684)
"e1684"
