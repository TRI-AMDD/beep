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

NEW_VER = datetime.datetime.today().strftime("%Y.%-m.%-d.%-H")


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
        f.write("\n")

    lines = []
    with open("setup.py", "rt") as f:
        for l in f:
            lines.append(re.sub(r'version=([^,]+),', 'version="%s",' % NEW_VER,
                                l.rstrip()))
    with open("setup.py", "wt") as f:
        f.write("\n".join(lines))
        f.write("\n")


@task
def merge_stable(ctx):
    """
    Tag and merge into stable branch.

    :param ctx:
    """
    ctx.run("git commit -a -m \"v%s release\"" % (NEW_VER, ), warn=True)
    ctx.run("git tag -a v%s -m \"v%s release\"" % (NEW_VER, NEW_VER))
    ctx.run("git push --tags")
    ctx.run("git checkout stable")
    ctx.run("git pull")
    ctx.run("git merge master")
    ctx.run("git push")
    ctx.run("git checkout master")


@task
def release_github(ctx):
    """
    Release to Github using Github API.

    :param ctx:
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
        headers={"Authorization": "token " + os.environ["GITHUB_RELEASES_TOKEN"]})
    print(response.text)


@task
def update_changelog(ctx):
    """
    Create a preliminary change log using the git logs.

    :param ctx:
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
    ctx.run("open CHANGES.md")


@task
def release(ctx, notest=False, nover=False):
    """
    Run full sequence for releasing beep.

    :param ctx:
    :param notest: Whether to skip tests.
    :param notest: Whether to skip autoversion (e. g. if tagging version).
    """
    ctx.run("rm -r dist build beep.egg-info", warn=True)
    if not nover:
        set_ver(ctx)
    if not notest:
        ctx.run("pytest beep")
    publish(ctx)
    merge_stable(ctx)
    release_github(ctx)
