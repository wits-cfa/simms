import os
from simms.skymodel.skydef import Catalogue
from scabha.basetypes import File
from simms import BIN

log = init_logger(BIN.skysim)

def simulate_catalogue(catalogue_file: File, delimiter: str = " "):
        catalogue = Catalogue(path=catalogue_file, delimiter=delimiter)
        catalogue.readfromfile()

        for index, source_data in enumerate(catalogue.sources, start=1):
            log.info(f"Source {index}:")
            for key, value in source_data.items():
                log.info(f"  {key}: {value}")

     
