echo 'export PATH="/etc/poetry/bin:$PATH"' >> ${HOME}/.bashrc
source ${HOME}/.bashrc

poetry config virtualenvs.create false

poetry config installer.max-workers 10
poetry install --no-interaction --no-ansi -vv