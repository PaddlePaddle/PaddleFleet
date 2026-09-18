# 仓库 Skills

所有 Skill 的实体文件统一维护在本目录。客户端通过相对软链接加载同一份内容：

| Skill | 实体 | Codex 入口 | Claude 入口 |
| --- | --- | --- | --- |
| `ai-review` | `skills/ai-review/` | `.agents/skills/ai-review` | `.claude/skills/ai-review` |
| `ci-analyzer` | `skills/ci-analyzer/` | `.agents/skills/ci-analyzer` | `.claude/skills/ci-analyzer` |

两个入口的链接目标均为 `../../skills/<name>`。保留相对路径，移动整个 checkout 后仍可解析。不要在入口目录维护副本，也不要把软链接替换成只含目标路径的普通文件。

新增 Skill 时，在 `skills/<name>/` 创建 `SKILL.md` 及所需资源，并从仓库根目录为两个客户端补充入口。以下以 `example-skill` 为例，使用 GNU/Linux 的 `ln -sT`，入口已存在时会报错：

```bash
ln -sT -- ../../skills/example-skill .agents/skills/example-skill
ln -sT -- ../../skills/example-skill .claude/skills/example-skill
```

将 `example-skill` 替换成实际技能名。若入口已存在，先检查目标，避免覆盖其他文件。技能内部使用相对于自身的资源路径。

验证实体和两端入口均可读取 `SKILL.md`、引用文件无死链，并运行仓库 `check-symlinks` 检查。`.claude/skills/` 纳入版本控制，其他 `.claude` 本地配置仍被忽略。
