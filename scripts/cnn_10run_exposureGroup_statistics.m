%% 

sound_levels = [10 20 30 40 50 60 70 80];

script_folder = fileparts(mfilename("fullpath"));
base_folder = fileparts(script_folder);

data_folder = fullfile(base_folder, "outputs", "dualChannel_data_forPaper");

ch1 = fullfile(data_folder, "ch1_10runs_20260623_144134", "ch1_all_trial_stats.csv");
ch2 = fullfile(data_folder, "ch2_10runs_20260623_160201", "ch2_all_trial_stats.csv");

run_stats(ch1, "Ch1", sound_levels)
run_stats(ch2, "Ch2", sound_levels)

%%

function run_stats(file, channel, sound_levels)

fprintf('\n %s \n', channel)

opts = detectImportOptions(file, TextType="string");

opts.SelectedVariableNames = {'run_number','signal_id','subject','group', 'sound_level','frequency','day','is_corr'};
opts = setvartype(opts, {'signal_id','subject','group','frequency','day','is_corr'}, 'string');

tbl = readtable(file, opts);

tbl.group = string(tbl.group);
tbl.is_corr = strcmpi(string(tbl.is_corr), "true");

tbl = tbl(ismember(tbl.group, ["Functional", "Sham"]) & ismember(tbl.sound_level, sound_levels), :);

tbl.subject = categorical(tbl.subject);
tbl.group = categorical(tbl.group, ["Functional", "Sham"]);
tbl.sound_level = categorical(string(tbl.sound_level), string(sound_levels));
tbl.frequency = categorical(tbl.frequency);
tbl.day = categorical(tbl.day);

sigTbl = groupsummary(tbl, {'signal_id','subject','group','sound_level','frequency','day'}, 'sum', 'is_corr');

sigTbl.Properties.VariableNames{'sum_is_corr'} = 'n_correct';
sigTbl.Properties.VariableNames{'GroupCount'} = 'n_total';


model_no_group = fitglme(sigTbl, 'n_correct ~ sound_level + (1|subject)', 'Distribution', 'Binomial', 'BinomialSize', sigTbl.n_total, ...
    'Link', 'logit', 'FitMethod', 'Laplace');

model_w_group = fitglme(sigTbl, 'n_correct ~ group + sound_level + (1|subject)', 'Distribution', 'Binomial', 'BinomialSize', sigTbl.n_total, ...
    'Link', 'logit', 'FitMethod', 'Laplace');

liklihood_rt = compare(model_no_group, model_w_group);
row = strcmp(model_w_group.CoefficientNames, 'group_Sham');

log_odds = model_w_group.Coefficients.Estimate(row);
log_odds_low = model_w_group.Coefficients.Lower(row);
log_odds_hi = model_w_group.Coefficients.Upper(row);

OR = exp(log_odds);
OR_low = exp(log_odds_low);
OR_hi = exp(log_odds_hi);

group_wald_p = model_w_group.Coefficients.pValue(row);
group_lrt_p = liklihood_rt.pValue(2);



model_no_group_2 = fitglme(sigTbl, 'n_correct ~ sound_level + frequency + day + (1|subject)', 'Distribution', 'Binomial', 'BinomialSize', sigTbl.n_total, ...
    'Link', 'logit', 'FitMethod', 'Laplace');

model_w_group_2 = fitglme(sigTbl, 'n_correct ~ group + sound_level + frequency + day + (1|subject)', 'Distribution', 'Binomial', 'BinomialSize', sigTbl.n_total, ...
    'Link', 'logit', 'FitMethod', 'Laplace');

liklihood_rt_2 = compare(model_no_group_2, model_w_group_2);
row_2 = strcmp(model_w_group_2.CoefficientNames, 'group_Sham');

log_odds_2 = model_w_group_2.Coefficients.Estimate(row_2);
log_odds_low_2 = model_w_group_2.Coefficients.Lower(row_2);
log_odds_hi_2 = model_w_group_2.Coefficients.Upper(row_2);

OR_2 = exp(log_odds_2);
OR_low_2 = exp(log_odds_low_2);
OR_hi_2 = exp(log_odds_hi_2);

group_wald_p_2 = model_w_group_2.Coefficients.pValue(row_2);
group_lrt_p_2 = liklihood_rt_2.pValue(2);


%observed accuracy
funct_acc = (sum(tbl.is_corr(tbl.group == "Functional")) / sum(tbl.group == "Functional")) * 100;
sham_acc = (sum(tbl.is_corr(tbl.group == "Sham")) / sum(tbl.group == "Sham")) * 100;



fprintf('Functional acc = %.2f%%\n', funct_acc)
fprintf('Sham acc = %.2f%%\n', sham_acc)

fprintf('\n primary \n')
fprintf('OR = %.3f\n', OR)
fprintf('95%% CI = %.3f - %.3f\n', OR_low, OR_hi)
fprintf('Wald p = %.4g\n', group_wald_p)
fprintf('LRT p = %.4g\n', group_lrt_p)

fprintf('\n secondary \n')
fprintf('OR = %.3f\n', OR_2)
fprintf('95%% CI = %.3f - %.3f\n', OR_low_2, OR_hi_2)
fprintf('Wald p = %.4g\n', group_wald_p_2)
fprintf('LRT p = %.4g\n', group_lrt_p_2)

end