# Import libraries
import pandas as pd
import folium
import os
import vincent
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import plotly.offline as py
import plotly.graph_objs as go

def plot_correlations(crude_corr, age_adj_corr):
    if crude_corr.shape[0]>30:
        f, ax = plt.subplots( figsize=(30, 10))
    else:
        f, ax = plt.subplots( figsize=(10, 10))
    
    sns.heatmap(crude_corr.transpose(), mask=np.zeros_like(crude_corr.transpose(), dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True).set_title('crude correlation')
    sns.heatmap(age_adj_corr.transpose(), mask=np.zeros_like(age_adj_corr.transpose(), dtype=np.bool),
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True).set_title('age adjusted correlation')
    return(f)

def graph(df,state):
    df_r = df[df['LocationAbbr'] == state]
    chart = alt.Chart(df_r).mark_bar().encode(
        x='Data_Value',
        y='Year',
    )
    return(chart)

def build_json_plot(df,prd, state):
    df_r = df[df['LocationAbbr'] == 'LA'].set_index('Year')
    dic = pd.concat([prd[[state]],df_r],axis=1)
    dic.index = np.arange(dic.shape[0])
    dic.columns = ['Prediction %s'%state,'LocationAbbr', 'True Value' ]+ list(dic.columns[3:])

    line = vincent.Line(dic[['True Value','Prediction %s'%state]], 
                        columns=['True Value','Prediction %s'%state],key_on='idx')
    line.axis_titles(x='Year', y='Diabete in '+state)
    line.legend(title='Diabete')
    line.height = 200
    line.width = 200
    return(line.grammar())    
 
def create_map(dataf,year):
    
    state_geo = os.path.join('BIN/data/us-states.json')
    
    with open('BIN/data/location_state.txt', 'r') as rd:
        f = rd.read()
        location = eval(f)
    
    with open('BIN/data/states.txt', 'r') as rd:
        f = rd.read()
        states = eval(f)
    prd = pd.read_csv('BIN/data/predictions_in_states.csv', index_col=0)
    # Initialize the map:
    m = folium.Map(location=[52, -112], zoom_start=3)
    df = pd.read_csv('BIN/data/scores.csv', index_col=0)
    # Add the color for the chloropleth:
    m.choropleth(
        geo_data=state_geo,
        name='choropleth',
        data=df,
        columns=['States', 'R^2'],
        fill_color='GnBu',
        key_on='feature.id',
        fill_opacity=0.8,
        line_opacity=0.6,
        legend_name='R^2 prediction'
    )
    folium.LayerControl().add_to(m)
    m.add_child(folium.LatLngPopup())
    for s in states:
        if s=='NH' or s=='WI' or s=='WV' or s=='DC':
            continue
        folium.RegularPolygonMarker(
            [location[s][0], location[s][1]],
            fill_color='#43d9de',
            radius=4,
            popup=folium.Popup(max_width=450).add_child(
            folium.Vega(build_json_plot(dataf,prd, state = s),width=250, height=250 ))
         ).add_to(m)
    
    m.save('BIN/data/map.html')

    return(m)



def create_a_3D_plot(G, st, labels):
    layt = nx.spring_layout(G, dim=3)
    # creation of coordinates
    Edges = list(G.edges)
    nodes = list(G.nodes)
    N = len(G)

    Xn=[layt[k][0] for k in G.nodes]# x-coordinates of nodes
    Yn=[layt[k][1] for k in G.nodes]# y-coordinates
    Zn=[layt[k][2] for k in G.nodes]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in Edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]

    # 3 D plot

    trace1=go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=go.Line(color='rgb(125,125,125)', width=1),
                   hoverinfo='none'
                   )
    trace2=go.Scatter3d(x=Xn,
                   y=Yn,
                   z=Zn,
                   mode='markers',
                   name='actors',
                   marker=go.Marker(symbol='dot',
                                 size=6,
                                 color=labels,
                                 colorscale=[[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']],
                                 line=go.Line(color='rgb(50,50,50)', width=0.5)
                                 ),
                    text=list(G.nodes),
                    hoverinfo='text'
                   )

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    layout = go.Layout(
             title="Network of %s (3D visualization)"%st,
             width=1000,
             height=1000,
             showlegend=False,
             scene=go.Scene(
             xaxis=go.XAxis(axis),
             yaxis=go.YAxis(axis),
             zaxis=go.ZAxis(axis),
            ),
         margin=go.Margin(
            t=100
        ),
        hovermode='closest'
       )

    data=go.Data([trace1, trace2])
    fig=go.Figure(data=data, layout=layout)

    py.iplot(fig, filename='3D graph')
    
    
    
    
def best_alpha(best_alpha, scores):
    ax = plt.subplot(111)
    t1 = range(len(scores))
    elm = [best_alpha, scores]
    st = ['best_alpha', 'scores']
    for l in [0,1]:
        plt.plot(t1, elm[l],label=st[l])

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)



    plt.show()
    
def plot_regression(y, prediction,score, score_2):
    fig, ax = plt.subplots()
    ax.scatter(y, prediction, edgecolors=(0, 0, 0))
    lo = [y.min(), y.max()]
    ax.plot(lo, lo, 'k--', lw=4)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.text(lo[0], lo[1], r'score: $R^2$ = %.3f   MSE = %.3f'%(score, score_2), fontsize=15)
    plt.show()
    
    
    
def plot_regression_state(X_,prd_final, y_cov, y_s, st):
    plt.plot(X_, prd_final, 'y', lw=3, zorder=9, label = 'prediction %s'%st)
    plt.fill_between(X_, prd_final - np.sqrt(np.diag(y_cov)),
                     prd_final + np.sqrt(np.diag(y_cov)),
                     alpha=0.3, color='y')
    plt.plot(X_, y_s,'r',lw=2,label = 'True')
    plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=True)
    plt.show()