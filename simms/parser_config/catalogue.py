import os
import click
from simms.skymodel.skydef import Catalogue
"""
'This comment will be deleted when merging.'

Since we have a Catalogue class in skydef, I thought 
what we need is a command line interface code for just Catalogue like Sphe did for skysim, telescope and observe.
So that we can check if we can read(at least, for now) a catalouge file.
Integrating the codes is one hell of a job. This might work if we do that.
"""

@click.command()
@click.argument('catalogue_file', type=click.Path(exists=True))
def simulate_catalogue(catalogue_file):
    try:
      
        file_exist = os.path.splitext(catalogue_file)[1].lower()

        if file_exist == '.txt':
            delimiter = None
        elif file_exist == '.csv':
            delimiter = ','
        else:
            delimiter = None
            click.echo(f"Warning: Unsupported file format '{file_exist}'. No delimiter will be used.")

        catalogue = Catalogue(path=catalogue_file, delimiter=delimiter)
        catalogue.readfromfile()

        for index, source_data in enumerate(catalogue.sources, start=1):
            click.echo(f"Source {index}:")
            for key, value in source_data.items():
                click.echo(f"  {key}: {value}")

    except FileNotFoundError:
        click.echo(f"Error: Catalogue file not found at '{catalogue_file}'")
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}")

if __name__ == '__main__': 
    simulate_catalogue()

 
""" the command line to run the simulate_catalogue function is:
poetry run simulate_catalogue path/to/catalogue_file.
still getting an error 'expected str, byte or os.pathlike object, no NoneType'
gave me a hope the code might work """






     
