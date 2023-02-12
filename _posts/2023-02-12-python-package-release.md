---
layout: post
title:  "Fully automated release process for Python packages"
description: "Using GitHub actions to have a fully controlled and automated release process"
date:   2023-02-12 15:56:19 -0100
categories:
- python
- ci
- python package
---

Maintaining a Python package puts you in front of the choice of how to handle the release/ci process. Working in teams adds another level of complexity to it: Everybody should follow certain rules so the whole release process works flawlessly. Even if we are only a two-man team and our project does not have a crazy-complicated release process, having this whole process automated can make things much easier. In this article, I want to introduce our release process based on GitHub [action](https://github.com/features/actions) and [apps](https://docs.github.com/en/developers/apps/getting-started-with-apps/about-apps). The code examples shows can be found in this [template project](https://github.com/chrislemke/python_template_project)

## The commit
Let's start at the beginning. Assuming somebody is creating a commit, e.g. adding a new feature. Since we are following the [Conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) guidelines the commit message could look like this `feat: add an awesome new feature`. To make sure, that the commit has a suitable name, the code is formatted correctly, etc we use [pre-commit hooks](https://github.com/chrislemke/python_template_project/blob/main/.pre-commit-config.yaml). Quite a lot of stuff is happening there. After code-checking with [Black](https://github.com/psf/black), [isort](https://github.com/PyCQA/isort), [pylint](https://github.com/PyCQA/pylint), etc. the pre-commit hooks also checks if the `poetry.lock` file - which manages die dependencies - is updated and conforms to the dependencies in the `pyproject.toml` file.

To not break the conventional commit guidelines the pull requests to develop need to have as the title the name of the commit. So in this example, it would be: `feat: add an awesome new feature`. But what if the user just renames the PR wrongly, not following the guidelines? For this, we have a [GitHub action](https://github.com/chrislemke/python_template_project/blob/main/.github/workflows/pr-title.yml) which checks the PR title. This helps enforce expressive commit names.

## `develop` to `main`
The project contains of two protected branches. `develop` and `main`. In `develop`, we collect the new features, bug fixes, etc. `main` represents the state of production. After the successful commit and a squash-merge - which is used to "hide" all the irrelevant commits, and after we collected the selection of features and fixes in the `develop` branch. We create a pull request merging `develop` into `main`.

Usually, the approval of this PR is just a formality. The same applies to the checks.
For merging, we want to use fast-forward merge. Unfortunately, GitHub does not provide this feature. So we decided to use [Mergify](https://mergify.com/) for it. It automatically merges the PR once it is approved and all checks have passed. The [`mergify.yml`](https://github.com/chrislemke/python_template_project/blob/main/.github/mergify.yml) file is used to configure it.

```yml
- name: Develop to main fast-forward merge
conditions:
  - and:
	  - base=main
	  - head=develop
	  - -conflict
actions:
  merge:
	method: fast-forward
```
Here we tell Mergify to make a fast-forward merge from `head` to `base` only if no merge conflicts occurred. 


## The release
We are nearly done with the complete release process. Just a few steps left. Right now all new features and bug fixes are in `develop` and `main`. So both branches point to the same commit. So far so good. Let's release. 

Automatically after the merge to the `main` branch the [release actions](https://github.com/chrislemke/python_template_project/blob/main/.github/workflows/release.yml) have started. 

### Release, please
We are using Google's [release please](https://github.com/google-github-actions/release-please-action) for creating a change log and raising the package version in the `pyproject.toml` file.  This release action creates a meaningful change log with the help of conventional commits. Check out the [sk-transformers change log](https://github.com/chrislemke/sk-transformers/blob/main/CHANGELOG.md) for an example.

### Deployment
As soon as the release action has been executed, the deployment action is started. Since we use [Poetry](https://python-poetry.org/) for dependency management we also use it for the deployment of the package. After installing it, the deployment process is pretty straightforward:

```sh
poetry config pypi-token.pypi ${{ secrets.PYPI_TOKEN }}
poetry publish --build
```

Nothing more happens here. Next up: deployment of the documentation.

### MkDocs
For automatically creating an documentation we use [MkDocs](https://www.mkdocs.org/), [mkdocstrings](https://mkdocstrings.github.io/) as well as [
Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) to create an easy-to-navigate and beautiful-looking documentation. 

### Another fast-forward merge
There is still an open question. What happened to the `CHANGELOG.md`? It was created by the release-please action. But where did this file ended up?
In the `main` branch. So now `main` is ahead of `develop` 🤔. To bring both branches again in sync we need to merge `main` into `develop`. This is again automatically done by using Mergify. 

This is it! The package has successfully deployed, `main` and `develop` are in sync and ready to continue working with them.  

## There is something more
As already mentioned, one reason for this automated process is the insight, that humans make mistakes. They give meaningless commit names like `...`, they accidentally use the wrong merge function on GitHub or they create a pull request to the wrong base branch. Using pre-commit hooks and Mergify already helped to avoid the first two human-errors. But what about the latter? For this, we created some GitHub actions ([`pr-check.yml`](https://github.com/chrislemke/python_template_project/blob/main/.github/workflows/pr-check.yml)) that close invalid pull requests and write a comment to let the user know about his misbehavior:

```
🚨 🚨 🚨
Merging the `${{ github.head_ref }}` branch into `main` is not allowed.
Only `develop` (and release branches) can be merged into `main`.

**This pull request will be closed**❗️
```

Et voilà! Problem solved. 

## Some final words
Setting up those different actions lettings they work together successfully, implementing Mergify, etc. Was definitely not very easy. Debugging CI-stuff is always a hassle. But it was worth it. Our project is now safe from human mistakes - at least most of them -  and the full automation is so comfortable that a whole release feels like a single commit.

Thanks for reading. I wish you all a great amount of success. 

