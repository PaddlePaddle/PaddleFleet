# 扩展 AI Review 评审规则

`ai-review` 将规则分为两层：

- `references/review-rules.md`：适用于多数 PaddleFleet 变更的基础规则。
- `references/rules/*.md`：面向特定模块、技术或风险场景的扩展规则。

评审时会先读取基础规则的相关章节，再扫描扩展规则的适用范围和触发条件，仅加载与当前变更匹配的文件。扩展规则按目录自动发现，不需要修改 `SKILL.md` 或维护额外注册表。

## 新增规则

1. 确认规则不是基础规则的重复表述。通用规则直接修改 `references/review-rules.md`，模块专用规则放入 `references/rules/`。
2. 在 `references/rules/` 下新增 kebab-case 命名的 Markdown 文件，例如 `auto-configurator.md` 或 `custom-operators.md`。
3. 在文件开头声明适用路径和触发条件，正文只保留可验证、可执行的检查项。
4. 检查规则不会要求错误的测试目录、平台能力或兼容性承诺，并补充规则来源。
5. 运行 skill 校验和仓库 pre-commit 检查。

## 文件模板

```markdown
# <规则名称>

- 适用路径：`src/example/**`、`tests/example/**`
- 触发条件：修改 <接口、配置或行为>
- 规则来源：<设计文档、缺陷、测试或维护者约定>

## <检查主题>

- 检查 <具体条件>，避免 <可观察的影响>。
- 修改 <行为> 时同步验证 <调用方、回退路径或测试>。
```

适用路径应尽量具体；无法用路径表达时，用触发条件描述配置、数据流或跨模块行为。规则来源应指向仓库文档、已确认缺陷或稳定约定，不写未经验证的经验结论。

## 编写要求

- 每条规则包含检查对象、触发条件和潜在影响，避免“注意性能”等泛化描述。
- 不在扩展文件中重复评论格式、问题优先级或发布策略，这些由评审调用方管理。
- 更具体的扩展规则优先于基础规则；发现冲突时先修正规则，不能让评审者自行猜测。
- 一个文件聚焦一个模块或主题；规则较多时按检查主题分节，避免评审加载无关内容。
- 测试要求应指向最接近变更模块的现有测试目录，并区分单卡、多卡和硬件相关场景。

## 验证

```bash
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  .agents/skills/ai-review

pre-commit run --files \
  .agents/skills/ai-review/SKILL.md \
  .agents/skills/ai-review/README.md \
  .agents/skills/ai-review/references/rules/<rule-name>.md
```

提交前用一个匹配适用范围的真实 diff 试评审，确认扩展规则会被加载；再用一个无关 diff 验证它不会进入评审上下文。
