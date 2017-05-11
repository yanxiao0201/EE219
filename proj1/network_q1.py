from matplotlib import pyplot as plt
from network_common import *

#Question number 1
def plot_size_versus_day():
#    results = []
#    plot_day = []
#    sums = [0, 0, 0, 0, 0]
#    days = 1
#    last_day = 'Monday'
#
#    for i in range(len(network_data)):
#        if network_data[i][1] != last_day:
#            last_day = network_data[i][1]
#            if days % 20 == 0:
#                results.append(sums)
#                sums = [0, 0, 0, 0, 0]
#                plot_day.append(days)
#            days += 1
#        
#        #update workflow
#        #find the index of the workflow
#        idx = np.argmax(workflow[i]) 
#        sums[idx] += float(network_data[i][5])
#    results.append(sums)
#    plot_day.append(days)
#
#    #plot size versus days
#    for k in range(5):
#        plt.figure(k)
#        plt.plot(plot_day, [row[k] for row in results], 'x')
#        plt.xlabel('days')
#        plt.xlim(1, 104)
#        #plt.ylim(210, 220)
#        plt.ylabel('size (GB)')
#    plt.show()

    results = []
    plot_day = []
    sums = [0, 0, 0, 0, 0]
    days = 1
    last_day = df['DAY'][0]

    for i in range(len(day_of_week)):
        if df['DAY'][i] != last_day:
            last_day = df['DAY'][i]
            if days % 20 == 0:
                results.append(sums)
                sums = [0, 0, 0, 0, 0]
                plot_day.append(days)
            days += 1
        
        #update workflow
        idx = workflow['work_flow_1'][i]+workflow['work_flow_2'][i]*2+workflow['work_flow_3'][i]*3+workflow['work_flow_4'][i]*4
        sums[idx] += size[i]

    results.append(sums)
    plot_day.append(days)

    lines = []
    #plot size versus days
    plt.figure('workflow')
    for k in range(5):
        line, = plt.plot(plot_day, [row[k] for row in results], 'o', label='workflow%d' %k)
        lines.append(line)
        plt.title('Workflow '+str(k)+' copy sizes versus time')
        plt.xlabel('days')
        plt.xlim(1, 104)
        #plt.ylim(210, 220)
        plt.ylabel('size (GB)')
    plt.legend(lines)
    plt.show()
