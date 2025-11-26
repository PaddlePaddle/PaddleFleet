# GitHub Copilot Custom Review Instructions

When reviewing code, focus on:

## Security Critical Issues
- Check for hardcoded secrets, API keys, or credentials
- Look for SQL injection and XSS vulnerabilities
- Verify proper input validation and sanitization
- Review authentication and authorization logic

## Performance Red Flags
- Identify N+1 database query problems
- Spot inefficient loops and algorithmic issues
- Check for memory leaks and resource cleanup
- Review caching opportunities for expensive operations

## Code Quality Essentials
- Functions should be focused and appropriately sized
- Use clear, descriptive naming conventions
- Ensure proper error handling throughout

## 🧪 单元测试鲁棒性与质量 (Unit Test Robustness)

当你审查测试文件（如 `test_*.py` 或 `*_test.cpp`）时，请严格检查以下几点：

* **边界条件覆盖：** 检查测试用例是否涵盖了所有关键的边界条件：**零值、负值、最大/最小限制、空输入 (e.g., `None`, 空列表/字典)、重复输入**、以及**无效的格式输入**。
* **异常路径验证：** 确保所有已知的失败路径和异常抛出都被测试覆盖。验证是否使用了适当的断言机制（如 `self.assertRaises` 或 `pytest.raises`）来捕获预期的错误。
* **测试隔离性：** 确认单元测试是自包含的。测试之间**禁止**存在依赖关系。如果需要外部资源（数据库、网络、文件系统），必须使用 **Mock 或 Stub** 进行完全隔离。
* **断言有效性与精度：** 确保断言是针对**业务逻辑的核心输出**，而不是无关的副作用。对于浮点数比较，必须使用具有容忍度的断言（如 `assertAlmostEqual`）。
* **不合理测试识别：** 标记任何冗余的断言或缺乏清晰目的的测试。所有测试名称必须清晰地描述它们正在验证的**特定场景和预期结果**。
* **参数化与可读性：** 对于具有相似逻辑的重复测试，建议使用参数化测试（Data-Driven Testing, DDT）来提高测试代码的可读性和维护性。
* **异步代码测试：** 对于 `async/await` 代码，确保测试函数本身是异步的，并且使用了正确的等待机制（如 `await` 或 `asyncio.run`）。

## Review Style
- Be specific and actionable in feedback
- Explain the "why" behind recommendations
- Acknowledge good patterns when you see them
- Ask clarifying questions when code intent is unclear

Always prioritize security vulnerabilities and performance issues that could impact users.

Always suggest changes to improve readability. For example, this suggestion seeks to make the code more readable and also makes the validation logic reusable and testable.

// Instead of:
if (user.email && user.email.includes('@') && user.email.length > 5) {
  submitButton.enabled = true;
} else {
  submitButton.enabled = false;
}

// Consider:
function isValidEmail(email) {
  return email && email.includes('@') && email.length > 5;
}

submitButton.enabled = isValidEmail(user.email);

## Description for pull request

- Please check the title of the Pull Request. It needs to follow the format of [CLASS]Title, for example, [BugFix] Fix memory leak of data processor. If the title is incorrect, provide suggestions on how the committer should modify it.
- Please check the description information of the Pull Request. At a minimum, it should explain why these modifications are being made in this Pull Request and what problem is being solved. If the committer hasn't written the corresponding information or the information is incomplete, prompt the committer to make modifications.

## Language
- For repository members, please reply directly in Simplified Chinese; for external Contributors, reply in English.
