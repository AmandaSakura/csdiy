# Git 与 GitHub 基础使用指南

本指南旨在提供一个清晰的、从零开始的 Git 工作流程，覆盖了从初始配置到日常使用的核心命令。目标是让你在需要时能快速回顾并上手使用 Git。

---

## 〇、首次使用前的全局配置 (只需一次)

在开始使用 Git 之前，你需要在你的设备上设置你的用户信息。这个信息会附加到你的每一次提交上。

```bash
# 设置你的用户名
git config --global user.name "你的名字"

# 设置你的邮箱地址
git config --global user.email "你的邮箱@example.com"
```

说明:

- `--global` 参数表示这是全局配置，这台设备上的所有 Git 仓库都会使用这个配置。
- 这个用户名和邮箱是你未来在提交代码时，展示的作者信息，建议设置为你 GitHub 的用户名和邮箱。

---

## 一、创建全新项目并关联到 GitHub

这个流程适用于你本地已经有了一个项目文件夹，现在想把它上传到 GitHub 并进行版本控制。

### 1. 在 GitHub 创建远程仓库

首先，访问 GitHub 网站，登录你的账户，然后创建一个新的仓库 (New repository)。

- 不要勾选 "Initialize this repository with a README"。因为我们本地将自行初始化。
- 创建成功后，复制仓库的 SSH 地址，格式为 `git@github.com:用户名/仓库名.git`。

### 2. 本地项目初始化并关联远程仓库

在你的本地项目文件夹中，打开终端或命令行工具，执行以下步骤。

```bash
# 1. 初始化本地仓库，这会在当前文件夹下创建一个 .git 目录
git init

# 2. 将本地所有文件添加到暂存区 (staging area)
# "." 代表当前目录下的所有文件和文件夹
git add .

# 3. 提交更改到本地仓库
# -m 后面是本次提交的说明，描述性越强越好
git commit -m "Initial commit: 项目初始化"

# 4. 将默认的 master 分支重命名为 main
# 这是为了与 GitHub 当前的默认分支名保持一致
git branch -M main

# 5. 添加远程仓库地址，并为其设置一个别名 "origin"
# 将下面的 URL 替换为你自己的仓库地址
git remote add origin git@github.com:AmandaSakura/csdiy.git
```

说明:

- `git init` 是开启 Git 版本控制的起点。
- `git add` 将文件的当前快照添加到了“暂存区”，这是一个准备提交的区域。
- `git commit` 则是将暂存区的内容正式保存到本地的版本历史中。
- `origin` 是远程仓库地址的通用别名，你也可以用别的名字，但 origin 是约定俗成的标准。

### 3. 推送本地内容到 GitHub

首次推送需要建立本地分支与远程分支的联系。

```bash
# -u 参数会关联本地的 main 分支和远程的 origin/main 分支
git push -u origin main
```

对 `-u` 参数的解释：`-u` (或 `--set-upstream`) 参数做了两件重要的事情：

- **推送内容**: 将你本地 main 分支上的所有提交推送到名为 origin 的远程仓库的 main 分支上。
- **建立跟踪关系**: 将本地 main 分支与远程 origin/main 分支关联起来。

建立这个“上游跟踪”关系后，未来在这个分支上，你就可以简化命令：

- 使用 `git push` 代替 `git push origin main`。
- 使用 `git pull` 代替 `git pull origin main`。

### 4. 验证

前往你的 GitHub 仓库页面，刷新一下，你应该能看到刚才推送的文件。至此，你的本地项目已成功与远程仓库关联。

---

## 二、日常开发核心流程

在你完成首次推送后，日常的开发和更新遵循以下循环。

### 1. 查看状态

在修改代码前后，随时可以使用此命令查看当前仓库的状态。

```bash
# 查看哪些文件被修改、哪些文件是新增的
git status
```

### 2. 添加与提交

对代码进行修改后，你需要再次 add 和 commit。

```bash
# 添加指定文件到暂存区
git add <文件名>

# 或者，添加所有修改和新增的文件
git add .

# 提交并撰写清晰的提交信息
git commit -m "feat: 添加了用户登录功能"
```

### 3. 推送更新

将本地的提交推送到远程仓库，与团队成员同步。

```bash
# 因为已经建立了跟踪关系，直接 push 即可
git push
```

关于 `git push -f`:

- `-f` 或 `--force` 是强制推送，它会用你本地的版本历史覆盖远程仓库的历史。
- **这是一个危险操作！** 只有在你非常确定需要这么做时（比如整理了个人分支的提交历史），并且是在个人分支上时，才可谨慎使用。**绝对不要对 main 或团队共享的分支使用强制推送！**

---

## 三、补充的重要 Git 概念与命令

为了能够完整地使用 Git，以下是一些必须掌握的补充知识。

### 1. 克隆现有仓库

如果你要参与一个已经存在的项目，第一步通常不是 init，而是 clone。

```bash
# 从 GitHub 克隆一个项目到你的本地
# 这会自动设置好 origin 和 main 分支的跟踪关系
git clone git@github.com:用户名/仓库名.git
```

### 2. 分支管理 (Branching)

分支是 Git 的核心功能，它允许你独立于主线 (main) 进行开发，避免了直接污染稳定代码。

```bash
# 查看所有本地分支，星号(*)标记的是当前所在分支
git branch

# 创建一个名为 "new-feature" 的新分支
git branch new-feature

# 切换到 "new-feature" 分支
git switch new-feature

# 或者使用老命令: git checkout new-feature

# (推荐) 创建并立即切换到新分支
git switch -c new-feature

# 在新分支上进行 add, commit, push 操作...

# 推送新分支到远程
git push -u origin new-feature

# 开发完成后，切换回主分支
git switch main

# 将 new-feature 分支的更改合并到 main 分支
git merge new-feature

# 删除已经不需要的本地分支
git branch -d new-feature
```

### 3. 同步远程仓库的更新

当团队其他成员向远程仓库推送了更新后，你需要将这些更新同步到你的本地。

```bash
# 从远程仓库拉取最新更改并自动与本地分支合并
# 这是 git fetch 和 git merge 的组合
git pull
```

### 4. 查看历史记录

```bash
# 查看提交日志
git log

# 查看更简洁的单行日志
git log --oneline

# 查看带分支图形的日志
git log --oneline --graph --all
```

### 5. 忽略文件 (.gitignore)

有些文件（如编译产物、日志文件、敏感配置）不应该被提交到版本库中。你可以在项目根目录下创建一个名为 `.gitignore` 的文件，并在其中列出要忽略的文件或目录模式。

**`.gitignore` 文件示例:**

```
# 忽略所有 .log 文件
*.log

# 忽略 node_modules 目录
/node_modules

# 忽略 IDE 配置文件
.idea/
.vscode/
```
