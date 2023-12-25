# README文件

关于 README.md 文件。

## 概述

`README.md` 文件是描述一个目录的 Markdown 文件。当你在 GitHub 和 Gitiles 中浏览这个目录时，就会展现这个文件。

例如，当你查看这个目录的内容时，就会展现它里面的 /README.md 文件：

https://github.com/google/styleguide/tree/gh-pages

在 Gitiles 中，当显示仓库库索引时，在 `HEAD` 引用中的 `README.md` 也会展现出来：

https://gerrit.googlesource.com/gitiles/

## 准则

**`README.md` 文件旨在为浏览您的代码的工程师（尤其是初次使用的用户）提供方向**。 `README.md` 文件可能是读者在浏览包含您的代码的目录时遇到的第一个文件。这种情况下，它也充当了目录的说明页面。

我们建议您的代码的顶级目录包含最新的`README.md`文件。这对于为其他团队提供接口的软件包目录尤其重要。

### 文件名

统一用 `README.md`。在 Gitiles 的目录视图中，不会显示名字叫 `README` 的文件。

### 内容

至少每个包级别的 `README.md` 都应当包含或指向以下信息：

1.  **此包/库中有什么**，它的用途是什么。
2.  **联系谁**。
3.  **状态**：此包/库是否已弃用，是否不用于一般发布等。
4.  **更多信息**：哪里能找到更详细的文档，例如：
     * overview.md 文件，提供更详细的概念信息。
     * 使用此软件包/库的任何 API 文档。