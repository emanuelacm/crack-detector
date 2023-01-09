import os
import pandas as pd
import folium as folium
import numpy as np
import datetime
from branca.element import Template, MacroElement
from branca.element import Element

#===========================================================
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import geopandas as gpd
import json
import re

#df_in = pd.read_csv('combinado_2.csv')

def generar_mapa(df,df_carac,save_name):
    clases = ['grieta longitudinal','grieta transversal','grieta cocodrilo']
    
    df_out = (df.groupby(['latitude','longitude'], as_index=False)
            .agg(**{'total_longitudinal': ('Tipo', lambda x: int(x[x == 0.0].size)),
            'total_transversal': ('Tipo', lambda x: int(x[x == 1.0].size)),
            'total_cocodrilo': ('Tipo', lambda x: int(x[x == 2.0].size)),                   
            'total_count': ('Tipo', lambda x: int(x[x == 0.0].size + x[x == 1.0].size + x[x == 2.0].size)),
            'grietas_ID':('ID', lambda x: ",".join(x.astype(str)))
            }))

    df_out.to_csv('cant_grietas.csv', encoding='utf-8', index=False)
    
    geojson = {"type": "FeatureCollection", "features": []}
    
    for i in list(range(len(df_out)-1)):
        row_in = pd.DataFrame()
        row_en = pd.DataFrame()
    
        row_in = df_out.loc[i]
        row_en = df_out.loc[i+1]
    
        feature = {"type": "Feature", 
                    "geometry": {"type": "LineString", 
                                 "coordinates": [
                                     [row_in['longitude'], row_in['latitude']],
                                     [row_en['longitude'], row_en['latitude']]
                                                 ]
                                 }, 
                    "properties": {"grietas":row_in['total_count'],
                                   "grietas_longitudinales":row_in['total_longitudinal'],
                                   "grietas_transversales":row_in['total_transversal'],
                                   "grietas_cocodrilo":row_in['total_cocodrilo'],
                                   "ID":row_in['grietas_ID'],
                                   "DATA":[]
                                   }
                    }
        ids = row_in['grietas_ID'].split(',')
        
        for id in ids:
            if(id != 'nan'):
                carac = df_carac.query('ID=='+ str(id)).to_numpy().flatten()

                data = {"IDg": id ,
                        "Tipo": clases[int(float(carac[1]))],
                        "Espesor": carac[2],
                        "Largo": carac[3],
                        "Path":carac[4]
                        }
            else:
                data = {"IDg": 'nan' ,
                        "Tipo": 'nan',
                        "Espesor": 'nan',
                        "Largo": 'nan',
                        "Path":"nan"
                        }
                
                
            feature['properties']['DATA'].append(data)
        

        geojson['features'].append(feature)
    
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)
    
    with open('result.geojson', 'w') as fp:
        json.dump(geojson, fp,cls=NpEncoder)    

    geoJSON_df = gpd.read_file('result.geojson')

    print(geoJSON_df)

    map = geoJSON_df.explore(
         column="grietas_longitudinales",
         tooltip="grietas_longitudinales",
         style_kwds=dict(weight=10),
         popup=True,
         cmap="Blues",
         name="Grietas longitudinales"
        )

    geoJSON_df.explore(
         m=map,
         column="grietas_transversales",
         tooltip="grietas_transversales",
         style_kwds=dict(weight=10),
         popup=True,
         cmap="Purples",
         name="Grietas transversales"
        )

    geoJSON_df.explore(
         m=map,
         column="grietas_cocodrilo",
         tooltip="grietas_cocodrilo",
         style_kwds=dict(weight=10),
         popup=True,
         cmap="Greens",
         name="Grietas cocodrilo"
        )

    geoJSON_df.explore(
         m=map,
         column="grietas",
         tooltip="grietas",
         style_kwds=dict(weight=10),
         popup=True,
         cmap="Oranges",
         name="Grietas"
        )
    

    folium.TileLayer('CartoDB positron').add_to(map)
    folium.TileLayer('openstreetmap').add_to(map) 
    folium.LayerControl().add_to(map)
        
    map.save(outfile = save_name)

    with open(save_name, "r", encoding='utf-8') as f:
        html = f.read()  

    between_script_tags = re.findall(r'(?<=geo_json_)[^_]+?[^\W_]*', html)

    between_script_tags = list(set(between_script_tags))

    for i in between_script_tags:  
        template = """  geo_json_""" + str(i) + """.bindPopup(
        function(layer){
        let div = L.DomUtil.create('div');
    
        let handleObject = feature=>typeof(feature)=='object' ? JSON.stringify(feature) : feature;
        let fields = ["grietas", "grietas_longitudinales", "grietas_transversales", "grietas_cocodrilo", "ID"];
        let aliases = ["grietas", "grietas_longitudinales", "grietas_transversales", "grietas_cocodrilo", "ID"];

        var value=0;
        var combo ='';
        var list_id = handleObject(layer.feature.properties['ID']).toLocaleString().split(",");
            
        if (list_id[0]==='nan') {
            var table =`<div class="grieta-head">
                            No hay grietas
                        </div>`
        } else {
            const tabla_data = JSON.parse(layer.feature.properties['DATA']);
        
            longitud = layer.feature.bbox[0];
            latitud = layer.feature.bbox[1];
            
            var value=0;
            var combo ='';

            list_id.forEach(function(id){
                combo += '<option data-tipo="'+ tabla_data[value].Tipo +'" data-espesor="' + tabla_data[value].Espesor + '" data-largo="' + tabla_data[value].Largo + '" data-path="' + tabla_data[value].
                Path+ '">'+ id +'</option>';
                value+=1;
            });

            var table = `
	        <div class="grieta-head">
	        <h3>Grieta</h3>
	        <p>Seleccione el ID de la grieta:
            <div class="box">
                <select id="grieta_sel" onchange="func_ema()">
	        	    ` + combo + `
	        	</select>
            </div>
            </p>
            </div>
	        <table class="table">
	        	<tr>
	        	<th>Tipo de grieta</th>
	        	<td data-cell > ` + tabla_data[0].Tipo + ` </td>
	        	</tr>
	        	<tr>
	        	<th>Espesor</th>
	        	<td data-cell >` + tabla_data[0].Espesor + `</td>
	        	</tr>
	        	<tr>
	        	<th>Largo</th>
	        	<td data-cell >` + tabla_data[0].Largo + `</td>
	        	</tr>
	        	<tr>
	        	<th>Latitud</th>
	        	<td>`+ latitud +`</td>
	        	</tr>
	        	<tr>
	        	<th>Longitud</th>
	        	<td>`+ longitud +`</td>
	        	</tr>
	        </table>
	        <h3>
            <div id="image">
	        	<img class="image" src="` + tabla_data[0].Path + `" alt="Grieta" width="300" height="300" style="object-fit:contain;"/>
            </div>
	        </h3>
            ` }
        div=table;
        return div
        },{"className": "foliumpopup"}); """

        map.get_root().script.add_child(Element(template))
    
    func_em = """ function func_ema(){
        val_cell =[$('#grieta_sel').find(':selected').data("tipo"), $('#grieta_sel').find(':selected').data("espesor"),$('#grieta_sel').find(':selected').data("largo"),$('#grieta_sel').find(':selected').data("path")]; 
    
        var cells = $('[class="table"] td[data-cell]');
        content=0;
        
        $(".image").attr("src",val_cell[3]);
    
        $(cells).each(function(){
            $(this).text(val_cell[content]);
            content+=1;
            });  
        }; 
        
        """
        
    map.get_root().script.add_child(Element(func_em))

    map.save(outfile = save_name)

    return map

