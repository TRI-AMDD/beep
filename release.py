"""
Pyinvoke tasks.py file for automating releases

This is a refactor of S. P. Ong's pymatgen administration
tasks.py, see https://github.com/materialsproject/pymatgen
"""

from invoke import task
import os
import json
import requests
import re
import subprocess
import datetime

from beep import __version__ as CURRENT_VER

NEW_VER = datetime.datetime.today().strftime("%Y.%-m.%-d")


@task
def publish(ctx):
    """
    Upload release to Pypi using twine.

    :param ctx:
    """
    ctx.run("rm dist/*.*", warn=True)
    ctx.run("python setup.py sdist bdist_wheel")
    ctx.run("twine upload dist/*")


@task
def set_ver(ctx):
    lines = []
    with open("beep/__init__.py", "rt") as f:
        for l in f:
            if l.startswith("__version__"):
                lines.append('__version__ = "%s"' % NEW_VER)
            else:
                lines.append(l.rstrip())
    with open("beep/__init__.py", "wt") as f:
        f.write("\n".join(lines))

    lines = []
    with open("setup.py", "rt") as f:
        for l in f:
            lines.append(re.sub(r'version=([^,]+),', 'version="%s",' % NEW_VER,
                                l.rstrip()))
    with open("setup.py", "wt") as f:
        f.write("\n".join(lines))


def tag_version():
    """
    Tag and merge into stable branch.
    """
    subprocess.call("git commit -a -m \"v%s release\"" % (NEW_VER, ))
    subprocess.call("git tag -a v%s -m \"v%s release\"" % (NEW_VER, NEW_VER))
    subprocess.call("git push --tags")


def release_github():
    """
    Release to Github using Github API.
    """
    with open("CHANGES.md") as f:
        contents = f.read()
    toks = re.split(r"\-+", contents)
    desc = toks[1].strip()
    toks = desc.split("\n")
    desc = "\n".join(toks[:-1]).strip()
    payload = {
        "tag_name": "v" + NEW_VER,
        "target_commitish": "master",
        "name": "v" + NEW_VER,
        "body": desc,
        "draft": False,
        "prerelease": False
    }
    response = requests.post(
        "https://api.github.com/repos/ToyotaResearchInstitute/beep/releases",
        data=json.dumps(payload),
        headers={"Authorization": "token " + GITHUB_RELEASES_TOKEN})
    print(response.text)


def update_changelog():
    """
    Create a preliminary change log using the git logs.
    """
    output = subprocess.check_output(["git", "log", "--pretty=format:%s",
                                      "v%s..HEAD" % CURRENT_VER])
    lines = ["* " + l for l in output.decode("utf-8").strip().split("\n")]
    with open("CHANGES.md") as f:
        contents = f.read()
    l = "=========="
    toks = contents.split(l)
    head = "\n\nv%s\n" % NEW_VER + "-" * (len(NEW_VER) + 1) + "\n"
    toks.insert(-1, head + "\n".join(lines))
    with open("CHANGES.md", "w") as f:
        f.write(toks[0] + l + "".join(toks[1:]))
    subprocess.call([EDITOR, "CHANGES.md"])


def release():
    """
    Run full sequence for releasing beep.
    """
    set_ver()
    update_changelog()
    # tag_version()
    # release_github()


if __name__ == "__main__":
    release()
