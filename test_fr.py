import matlab.engine
eng = matlab.engine.start_matlab()
eng.run(r'D:\Desktop\Common\AdaptativeConvolution\MetricCode\Demo2_Full_Resolution_multi.m', nargout=0)
eng.quit()
print('\n\n\n')
file_path = r'D:\Desktop\Common\AdaptativeConvolution\MetricCode\test_wv3_OrigScale_multiExm1_Avg_FR_Assessment.tex'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
    last_three_lines = lines[-3:]
for line in last_three_lines:
    print(line.strip())
