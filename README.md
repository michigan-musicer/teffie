For a fuller description of the Teffie project, [read this blog post here](https://musicer-kw.com/software/teffie-ai-for-tefl/).

## Usage

Requires Python 3.10. Note that the `src/main.py` script needs to be run from the project root directory.

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt

bash run.sh
```

Note that the requirements.txt is a `pip3` freeze dump, which should be cleaned up to list actual top-level dependencies.

## Structure

`src/main.py` is the main driver program. Over time, the plan is to refactor a lot of hard-coded info from the driver file to other files for the sake of organization.
