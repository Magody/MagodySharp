using System;
using System.Collections;
using System.Collections.Generic;

using types;

using System.IO;
using System.Text.Json;

namespace custom_lib
{
    public class SerializationJSON
    {
        
        public static string saveJSON(Hashtable hashtable, string path_file="temp.json", bool save=false){
            JsonSerializerOptions options = new JsonSerializerOptions();
            options.WriteIndented = true;
            string json = JsonSerializer.Serialize<Hashtable>(hashtable, options);;
            if(save || !path_file.Equals("temp.json")){
                File.WriteAllText(path_file, json);
            }
            return json;
            
        }

        public static JsonElement loadJSON(string path_file){
            string jsonString = File.ReadAllText(path_file);
            return JsonSerializer.Deserialize<JsonElement>(jsonString)!;

        }

    }
}
