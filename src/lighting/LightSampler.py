import mathutils
import bpy
import random

from src.lighting.LightModule import LightModule
from src.utility.Utility import Utility
from src.utility.BoundingBoxSampler import BoundingBoxSampler
from src.utility.SphereSampler import SphereSampler


class LightSampler(LightModule):
    """ Samples light source\'s settings and sets them.
    
    **Configuration**:
    
    .. csv-table::
       :header: "Parameter", "Description"

       "known_sampling_methods", "A dict that where {key:value} pairs are {known sampling method:expected amount of parameters for this sampling method}."

    **Properties per sampling method**:

    .. csv-table::
       :header: "Keyword", "Description"

       "BoundingBoxSampling", "Uniform 3D sampling method that samples a random position inside a bounding box. See src/utility/BoundingBoxSampler.py for more info."
       "SphereSampling", "Samples a point in and inside a solid sphere. See src/utility/SphereSampler.py for more info."
    """

    def __init__(self, config):
        LightModule.__init__(self, config)
        self.known_sampling_methods = {
            "BoundingBoxSampler": 6,
            "SphereSampler": 4
        }

    def run(self):
        """ Sets light sources.

        """
        self.cross_source_settings = self._sample_settings(self.cross_source_settings)
        source_specs = self.config.get_list("lights", [])
        for i, source_spec in enumerate(source_specs): 
            # Create light data, link it to the new object
            light_data = bpy.data.lights.new(name="light_" + str(i), type=self.default_source_settings["type"])
            light_obj = bpy.data.objects.new(name="light_" + str(i), object_data=light_data) 
            # Set default light source
            self._init_default_light_source(light_data, light_obj)
            # Sample settings as specified in the config file
            sampled_settings = self._sample_settings(source_spec) 
            # Configure default light source
            self._set_light_source_from_config(light_data, light_obj, sampled_settings)
            bpy.context.collection.objects.link(light_obj)
            # Write settings to file
            self._write_settings_to_file(sampled_settings, self.config.get_string("path", ""))
            
    def _sample_settings(self, source_spec):
        """ Samples the parameters according to user-defined sampling types in the configration file.

        :param source_spec: Dict that contains settings defined in the config file.
        :return: Processed settings dict.
        """
        sampled_settings = {}
        for attribute_name, value in source_spec.items():
            # Check if settings value must be sampled
            if isinstance(value, dict):
                # Check if sampling type is specified
                if 'type' in value:
                    # Check if specified type is a known sampling type
                    if value['type'] in self.known_sampling_methods:

                        if not isinstance(value['sampling_params'], list):
                            sampling_params= [value['sampling_params']]
                        else:
                            sampling_params = value['sampling_params']

                        # Check if amount of values is correct
                        if len(sampling_params) != self.known_sampling_methods[value['type']]:
                            raise Exception("Wrong amount of arguments for " + value['type'])

                        if value['type'] == 'BoundingBoxSampler':
                            # Use BoundingBoxSampler
                            result = list(BoundingBoxSampler.sample(mathutils.Vector((sampling_params[0], sampling_params[1], sampling_params[2])), mathutils.Vector((sampling_params[3], sampling_params[4], sampling_params[5]))))
                            # Update output dict
                            sampled_settings.update({attribute_name : result})
                        elif value['type'] == 'SphereSampler':
                            # Use SphereSampler
                            result = list(SphereSampler.sample(mathutils.Vector((sampling_params[0], sampling_params[1], sampling_params[2])), sampling_params[3]))
                            # Update output dict
                            sampled_settings.update({attribute_name : result})
                    else:
                        raise Exception("Unknown sampling method: " + value['type'])
                else:
                    raise Exception("Samplimg method type not specified!")
            else:
                sampled_settings.update({attribute_name : value})

        return sampled_settings

    def _write_settings_to_file(self, sampled_settings, path):
        """ Writes light source settings used in a file.

        One light source is one line in the file, settings sorted alphabetically.

        :param sampled_settings: Dict with used settings where {key:value} pairs are {setting name: used setting value}.
        :param path: Path to output file specified in the configuration file.
        """
        line = ""
        # Sort merged source specific and cross source settings alphabetically
        settings_to_write = sorted(Utility.merge_dicts(sampled_settings, self.cross_source_settings).items(), key=lambda x: x[0])
        with open(Utility.resolve_path(path), 'a') as f:
            for item, value in settings_to_write:
                if not isinstance(value, list):
                    value = [value]
                line = line + " ".join(str(x) for x in value) + " "
            f.write('%s \n' % line)
    
