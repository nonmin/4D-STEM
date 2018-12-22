import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import row, widgetbox,gridplot,column,layout
from bokeh.models import ColumnDataSource, SaveTool,HoverTool, TapTool, CategoricalColorMapper, LinearColorMapper
from bokeh.models.widgets import Slider, TextInput, Button
from bokeh.plotting import figure
from bokeh.palettes import d3,Greys256,Inferno256,brewer
label_color_mapper=CategoricalColorMapper(factors=['-1','0', '1', '2', '3', '4', '5', '6', '7','8','9'],
                                          palette=[d3['Category20'][11][0],d3['Category20'][11][1],d3['Category20'][11][2],d3['Category20'][11][3],
                                                   d3['Category20'][11][4],d3['Category20'][11][5],d3['Category20'][11][6],
                                                   d3['Category20'][11][7],d3['Category20'][11][8], d3['Category20'][11][9],
                                                   d3['Category20'][11][10]])


intensity_color_mapper=LinearColorMapper(palette=Greys256)

ronch_color_mapper=LinearColorMapper(palette=Inferno256)#,high=9800,low=8400)

Ronch = np.load(r'expr_all_ronch.npy') 
Haadf = np.load(r'expr_haadf.npy').flatten()
Mean_overall_ronch = Ronch.mean(axis=0)
root_Coord = np.load(r'expr_ue_root.npy')
boot_Coord = np.load(r'expr_ue_1b.npy')
Label = np.load(r'expr_ue_1b_label.npy').flatten()
Label_str = ["%i" % x for x in Label]

nc = 11
# Image size
M = 64 #row
N = 64 #col

# Ronch size
u = 180 #row
v = 180 #col

#size_scale
size_scale=1

#Image pixel patch left-top corners (locations)
Image_pixel_sy, Image_pixel_sx = divmod(np.arange(M*N),N) 
Image_pixel_sy = -Image_pixel_sy                               

Ronch_pixel_sx, Ronch_pixel_sy = divmod(np.arange(u*v),v)

#Counter-clock wise
y_offsets=[0,-1,-1,0]
x_offsets=[0,0,1,1]
Image_patch_ys=[y_offsets+yy for yy in Image_pixel_sy]
Image_patch_xs=[x_offsets+xx for xx in Image_pixel_sx]
Ronch_patch_ys=[y_offsets+yy for yy in Ronch_pixel_sy]
Ronch_patch_xs=[x_offsets+xx for xx in Ronch_pixel_sx]


source_haddf = ColumnDataSource(data=dict(xs=Image_patch_xs,ys=Image_patch_ys,x=Image_pixel_sx,y=-Image_pixel_sy,labels=Label,intensity=Haadf))
source_image_patch_selected=ColumnDataSource(data=dict(xs=[],ys=[],color=[]))
source_image_patch_tapped=ColumnDataSource(data=dict(xs=[],ys=[]))

source_mean_ronch=ColumnDataSource(data=dict(xs=Ronch_patch_xs,ys=Ronch_patch_ys,intensity=[]))
source_tapped_ronch=ColumnDataSource(data=dict(xs=Ronch_patch_xs,ys=Ronch_patch_ys,intensity=[]))

source_maniCo=ColumnDataSource(data=dict(root_Co_x= root_Coord[:,0], root_Co_y= root_Coord[:,1], 
							   boot_Co_x= boot_Coord[:,0], boot_Co_y= boot_Coord[:,1], 
							   color = Label_str,Co_label= Label))

"""
UMAP Manifold Distribution
"""                                  
umap_plot=figure(width=64,height=64,
              tools="save, pan, box_zoom, wheel_zoom, box_select,lasso_select",
              toolbar_location="right")
umap_plot.axis.visible= False
umap_plot.title.text="UMAP Manifold Layout"
umap_plot.title.align='left'
umap_plot.title.text_font_size='15px'

tooltips_CM=[("cluster_label","@Co_label"),]
umap_plot.add_tools(HoverTool(tooltips=tooltips_CM))

umap=umap_plot.circle(x='root_Co_x',y='root_Co_y', 
                fill_color={'field': 'color', 'transform': label_color_mapper}, 
                legend = 'Co_label', fill_alpha=1,
                source=source_maniCo)
umap_plot.legend.orientation = "horizontal"
umap_plot.legend.location = "top_left"

"""
"Bootstrapped" UMAP Manifold Distribution
"""                                  
bumap_plot=figure(width=64,height=64,
              tools="save, pan, box_zoom, wheel_zoom, box_select,lasso_select",
             toolbar_location="right")
bumap_plot.axis.visible= False
bumap_plot.title.text="Bootstrapped UMAP Manifold Layout"
bumap_plot.title.align='left'
bumap_plot.title.text_font_size='15px'

tooltips_CM=[("cluster_label","@Co_label"),]
bumap_plot.add_tools(HoverTool(tooltips=tooltips_CM))

bumap=bumap_plot.circle(x='boot_Co_x',y='boot_Co_y', 
                fill_color={'field': 'color', 'transform': label_color_mapper}, 
                legend = 'Co_label', fill_alpha=1,
                source=source_maniCo)
bumap_plot.legend.orientation = "horizontal"
bumap_plot.legend.location = "top_left"

"""
Choosen Cluster Spatial Distribution
"""
Choosen_label_plot=figure(x_range=(0, N),y_range=(-M,0),
                          tools="save,pan,box_zoom, wheel_zoom, box_select,lasso_select",
                         toolbar_location="right")
Choosen_label_plot.axis.visible=False
Choosen_label_plot.grid.grid_line_color=None
Choosen_label_plot.title.text="Spatial distribution of selected cluster "
Choosen_label_plot.title.align='left'
Choosen_label_plot.title.text_font_size='15px'

#Background patches
CL_b=Choosen_label_plot.patches('xs','ys',
                            selection_fill_color={'field': 'intensity', 'transform': intensity_color_mapper},
           					# set visual properties for non-selected glyphs
               				nonselection_fill_alpha=1,
               				nonselection_fill_color={'field': 'intensity', 'transform': intensity_color_mapper},
                            fill_alpha=1,line_color="black",line_width=0,
                            source=source_haddf)
CL_b=Choosen_label_plot.patches('xs','ys', 
                                fill_color={'field': 'intensity', 'transform': intensity_color_mapper}, 
                                fill_alpha=0.5,line_color="black",line_width=0,
                                source=source_haddf)

tooltips_CL_b=[("location","(@x,@y)"),("cluster_label","@labels")]
Choosen_label_plot.add_tools(HoverTool(tooltips=tooltips_CL_b,renderers=[CL_b]))

Choosen_label_plot.add_tools(TapTool(renderers=[CL_b]))                   

#Selected patches
Choosen_label_plot.patches('xs','ys', fill_color={'field': 'color', 'transform': label_color_mapper}, fill_alpha=1,line_color="black",line_width=0,
                           source=source_image_patch_selected)

Choosen_label_plot.patches('xs','ys',
                   fill_alpha=0,line_width=3,line_color='Cyan',
                   source=source_image_patch_tapped)


"""
Cluster Mean Ronch Plot
"""
Cluster_ronch=figure(x_range=(0, v/size_scale),y_range=(0,u/size_scale),
                     tools="pan,save,box_zoom,wheel_zoom",
                    toolbar_location="right")
Cluster_ronch.grid.grid_line_color=None
Cluster_ronch.axis.visible=False
Cluster_ronch.title.text="Mean Ronch of selected cluster"
Cluster_ronch.title.align='left'
Cluster_ronch.title.text_font_size='15px'

Cluster_ronch.patches('xs','ys', 
                      fill_color={'field': 'intensity', 'transform': ronch_color_mapper} , 
                      fill_alpha=1,line_alpha=0,line_width=0,
                      source=source_mean_ronch)


"""
Tapped Ronch Plot
"""
Tapped_ronch=figure(x_range=(0,v/size_scale),y_range=(0,u/size_scale),
                    tools="save",toolbar_location="right")
Tapped_ronch.axis.visible=False
Tapped_ronch.grid.grid_line_color=None
Tapped_ronch.title.text="Ronch of tapped pixel"
Tapped_ronch.title.align='left'
Tapped_ronch.title.text_font_size='15px'

Tapped_ronch.patches('xs','ys', 
                    fill_color={'field': 'intensity', 'transform': ronch_color_mapper} , 
                    fill_alpha=1,line_alpha=0,line_width=0,source=source_tapped_ronch)


def update_Multiple_selection(attrname,old,new):
    """
    Update FT of tapped pixel on CL plot 
    """
    inds=np.array(new['1d']['indices']).flatten()
    if len(inds)==0:
        source_image_patch_selected.data=dict(xs=[],ys=[],color=[])
        source_mean_ronch.data['intensity']=[] #clean mean intensity
    else:
        image_x=Image_pixel_sx[inds]
        image_y=Image_pixel_sy[inds]    
        image_patch_ys=[y_offsets+yy for yy in image_y]
        image_patch_xs=[x_offsets+xx for xx in image_x]
        p_color = Label[inds]
        source_image_patch_selected.data=dict(xs=image_patch_xs,ys=image_patch_ys, 
        									  color=["%i" % x for x in p_color])    
        
        r_mean=Ronch[inds,:].mean(axis=0).flatten()
        r_mean = r_mean-Mean_overall_ronch
        source_mean_ronch.data['intensity']=r_mean
        
umap.data_source.on_change('selected',update_Multiple_selection)
bumap.data_source.on_change('selected',update_Multiple_selection)


def update_Tapped_CL(attrname,old,new):
    """
    Update Ronch of tapped pixel on CL plot 
    """
    inds=np.array(new['1d']['indices']).flatten()
    if len(inds)==0:
        source_tapped_ronch.data['intensity']=[] # clear intensity of tapped Ronch
        Tapped_ronch.title.text="Ronch of tapped pixel" # default title 
        source_image_patch_tapped.data=dict(xs=[],ys=[])                            
    else:    
        label_0 = Label[inds[0]]
        y_lol_0,x_lol_0=divmod(inds[0],N)
        ys_0=[y_offsets-yy for yy in np.asarray([y_lol_0])]
        xs_0=[x_offsets+xx for xx in np.asarray([x_lol_0])]
        t = Ronch[inds[0],:].flatten() 
        t = t-Mean_overall_ronch      
        source_tapped_ronch.data['intensity'] = t # updaate intensity of tapped original Ronch
        Tapped_ronch.title.text = "Ronch of tapped pixel at (%i,%i), cluster label: (%i)"%(x_lol_0,y_lol_0,label_0) # upate first tapped Ronch plot title
        source_image_patch_tapped.data=dict(xs=xs_0,ys=ys_0) 
                

CL_b.data_source.on_change('selected',update_Tapped_CL)
    

l=gridplot([[umap_plot,bumap_plot],
            [Choosen_label_plot,Cluster_ronch,Tapped_ronch]],sizing_mode='scale_width')
curdoc().add_root(l)
curdoc().title='Ronch Rover'