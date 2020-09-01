
def current_git_hash():
    repo = git.Repo(path=os.path.dirname(__file__), search_parent_directories=True)
    return repo.head.object.hexsha