const { GoogleGenAI } = require('@google/genai');

/**
 * Helper to get the GenAI client if GEMINI_API_KEY is configured
 */
function getGenAIClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey || apiKey.trim() === '' || apiKey.includes('your_gemini_api_key') || !apiKey.startsWith('AIzaSy')) {
    return null;
  }
  return new GoogleGenAI({ apiKey });
}

/**
 * 1. AI Task Breakdown: Deconstruct a task into structured steps, criteria & suggestions
 */
async function breakdownTask({ title, description, priority }) {
  const ai = getGenAIClient();

  if (!ai) {
    return getMockTaskBreakdown(title, description, priority);
  }

  const prompt = `You are an expert agile project manager and productivity assistant.
Analyze this task and break it down into an actionable, professional plan.

Task Title: "${title}"
Current Description: "${description || 'None provided'}"
Current Priority: "${priority || 'Medium'}"

Respond ONLY with valid JSON (no markdown formatting, no code fences):
{
  "enhancedTitle": "Refined and clear action-oriented title",
  "summary": "Clear 2-sentence overview of the objective",
  "suggestedPriority": "High | Medium | Low",
  "subtasks": [
    "Step 1: Actionable step with clear outcome",
    "Step 2: Actionable step with clear outcome",
    "Step 3: Actionable step with clear outcome",
    "Step 4: Actionable step with clear outcome"
  ],
  "estimatedTime": "e.g., 2-3 hours",
  "acceptanceCriteria": "Clear condition that marks this task as successfully done",
  "proTip": "One high-value practical tip to succeed faster"
}`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-flash-latest',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const responseText = response.text ? response.text.trim() : '';
    const cleaned = responseText.replace(/^```json\s*/i, '').replace(/```$/i, '').trim();
    return JSON.parse(cleaned);
  } catch (err) {
    console.warn(`[GeminiService] Live API call failed (${err.message}). Using fallback engine.`);
    return getMockTaskBreakdown(title, description, priority);
  }
}

/**
 * Helper to parse explicit calendar dates like "25th Nov", "Nov 25", "25 November"
 */
function parseExplicitDate(text) {
  if (!text || typeof text !== 'string') return null;

  const months = { jan: 1, feb: 2, mar: 3, apr: 4, may: 5, jun: 6, jul: 7, aug: 8, sep: 9, oct: 10, nov: 11, dec: 12 };
  const pad = (n) => String(n).padStart(2, '0');
  const now = new Date();
  const year = now.getFullYear();

  // Pattern 1: "25th Nov", "25 November", "on 25th Nov"
  const p1 = /\b(?:on|by|at)?\s*(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+(\d{4}))?\b/i;
  let m = text.match(p1);
  if (m) {
    const day = parseInt(m[1], 10);
    const month = months[m[2].toLowerCase().slice(0, 3)];
    const yr = m[3] ? parseInt(m[3], 10) : year;
    return `${yr}-${pad(month)}-${pad(day)}`;
  }

  // Pattern 2: "Nov 25th", "November 25", "on Nov 25"
  const p2 = /\b(?:on|by|at)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(\d{4}))?\b/i;
  m = text.match(p2);
  if (m) {
    const month = months[m[1].toLowerCase().slice(0, 3)];
    const day = parseInt(m[2], 10);
    const yr = m[3] ? parseInt(m[3], 10) : year;
    return `${yr}-${pad(month)}-${pad(day)}`;
  }

  return null;
}

/**
 * Helper to clean and format action titles, stripping out metadata tokens (priority, status, dates)
 */
function cleanTaskTitle(rawText) {
  if (!rawText || typeof rawText !== 'string') return 'Task Deliverable';

  let cleaned = rawText.trim();

  // 1. Normalize common typos and shortcuts
  cleaned = cleaned
    .replace(/\b(tom+ar+ow|tom+or+ow|tom+or+o|tmrw|tmw)\b/gi, 'tomorrow')
    .replace(/\b(todai|2day)\b/gi, 'today')
    .replace(/\b(priorty|prio)\b/gi, 'priority')
    .replace(/\b(progres+|in-progress)\b/gi, 'in progress')
    .replace(/\b(pend?ing)\b/gi, 'pending')
    .replace(/\b(compl?eted?)\b/gi, 'completed');

  // 2. Strip priority tokens and phrases
  cleaned = cleaned.replace(/\b(?:priority\s*:?\s*(?:high|medium|low|urgent|critical|normal)|(?:high|medium|low|urgent|critical|normal)\s*priority)\b/gi, '');
  cleaned = cleaned.replace(/\b(?:priority\s+(?:is\s+)?(?:high|medium|low))\b/gi, '');
  cleaned = cleaned.replace(/\b(urgent|critical|asap|p[0-3])\b/gi, '');
  cleaned = cleaned.replace(/\bpriority\b/gi, '');

  // 3. Strip status tokens and phrases
  cleaned = cleaned.replace(/\b(?:status\s*:?\s*)?(?:in[- ]progress|wip|doing|working on|actively working|completed|done|finished|pending|todo|to[- ]do)\b/gi, '');
  cleaned = cleaned.replace(/\bstatus\b/gi, '');

  // 4. Strip explicit dates like "on 25th Nov", "by Nov 30", "25 November"
  cleaned = cleaned.replace(/\b(?:on|by|at|due|for)?\s*\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:\s+\d{4})?\b/gi, '');
  cleaned = cleaned.replace(/\b(?:on|by|at|due|for)?\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:\s+\d{4})?\b/gi, '');

  // 5. Strip relative date tokens
  cleaned = cleaned.replace(/\b(?:due\s*(?:date|on|by)?\s*:?\s*)?(?:today|tonight|tomorrow|yesterday|eod|end of day)\b/gi, '');
  cleaned = cleaned.replace(/\b(?:by|on|before|until|for)\s+(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next\s+week|this\s+week|eod)\b/gi, '');
  cleaned = cleaned.replace(/\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b/gi, '');
  cleaned = cleaned.replace(/\b(?:next\s+week|this\s+week)\b/gi, '');

  // 6. Strip dangling prepositions, conjunctions, punctuation, and leading prefixes
  cleaned = cleaned
    .replace(/\s+(?:by|on|at|due|with|for|in|to|and|is|as)\s*$/gi, '')
    .replace(/^(?:task|todo|to-do|reminder):\s*/gi, '')
    .replace(/^[,\s;:\-\/]+|[,\s;:\-\/]+$/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  if (!cleaned) cleaned = rawText.trim();

  // 7. Convert to polished Title Case
  const minorWords = new Set(['a', 'an', 'the', 'and', 'but', 'or', 'for', 'nor', 'on', 'at', 'to', 'from', 'by', 'in', 'of']);
  const words = cleaned.split(' ');
  const formattedWords = words.map((word, idx) => {
    const lower = word.toLowerCase();
    if (idx > 0 && minorWords.has(lower)) {
      return lower;
    }
    if (/^(k8s|aws|api|ui|ux|rest|crud|jwt|sql|db|ci\/cd|pr|hpa|pvc|iam|yaml|json)$/i.test(word)) {
      return word.toUpperCase();
    }
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  });

  return formattedWords.join(' ');
}

/**
 * Generate natural, real human-like descriptions without robotic corporate buzzwords
 */
function generateHumanDescription(title, rawInput) {
  const t = (title || '').toLowerCase();
  const raw = (rawInput || '').toLowerCase();

  // 1. Sports, Fitness, Outdoor, Games (Cricket, Gym, Football, etc.)
  if (/cricket|football|soccer|gym|workout|badminton|run|running|exercise|swim|swimming|game|match|hike|trek|cycling|tennis|sports|practice/i.test(t) ||
      /cricket|football|soccer|gym|workout/i.test(raw)) {
    const sportName = t.includes('cricket') ? 'a cricket match' : t.toLowerCase();
    const summary = `Head out for ${sportName}. Get the gear ready, coordinate with friends, and enjoy the game.`;
    const steps = [
      `Pack sports kit, gear, and a water bottle`,
      `Confirm match timing and location with friends / team`,
      `Warm up and stretch before starting`,
      `Play the game and wrap up after`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 2. Capstone / Thesis / Major Academic Submission
  if (/capstone|thesis|dissertation|viva|final.*project/i.test(t) || /capstone|thesis/i.test(raw)) {
    const summary = `Finalize and submit the capstone project. Double-check all documentation, verify that the project runs cleanly, and ensure everything is ready for review.`;
    const steps = [
      `Review project checklist and rubric requirements`,
      `Run test cases and verify the app and containers work smoothly`,
      `Finalize project report, screenshots, and architecture diagrams`,
      `Upload all deliverables and submit before the deadline`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 3. Project / Assignment / Homework / General Submission
  if (/submit|assignment|homework|project|presentation|report|deck|slides/i.test(t)) {
    const summary = `Wrap up and submit ${title}. Review all requirements, verify deliverables, and make sure everything is turned in on time.`;
    const steps = [
      `Double-check assignment instructions and checklist`,
      `Review all files, code, or report sections for any mistakes`,
      `Save final changes and create a backup`,
      `Submit on the portal and confirm successful upload`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 4. Docker / Kubernetes / DevOps / Cloud / Deployment
  if (/docker|k8s|kubernetes|container|pipeline|jenkins|terraform|ansible|deploy|cloud|aws|server|cluster/i.test(t)) {
    const summary = `Set up and verify ${title}. Test everything locally first and make sure all services start cleanly without errors.`;
    const steps = [
      `Review configuration and required environment variables`,
      `Build and run containers / manifests to verify behavior`,
      `Check logs and test the health endpoints`,
      `Commit clean changes and document quick run instructions`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 5. Bug Fix / Coding / Development
  if (/fix|bug|issue|error|feature|code|api|endpoint|auth|frontend|backend/i.test(t)) {
    const summary = `Work on ${title}. Reproduce any issues, make the necessary code updates, and test to confirm it works properly.`;
    const steps = [
      `Check the error logs or reproduce the issue locally`,
      `Implement the fix or feature cleanly`,
      `Test to make sure nothing else broke`,
      `Commit changes and push the update`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 6. Study / Learning / DSA / LeetCode
  if (/study|learn|read|book|course|leetcode|practice|dsa|problem|tutorial/i.test(t)) {
    const summary = `Dedicate focused study time for ${title}. Work through the concepts and take notes on key takeaways.`;
    const steps = [
      `Go through the core concepts or problem statements`,
      `Solve the problems or practice hands-on examples`,
      `Note down key formulas or algorithmic patterns`,
      `Quick recap to lock in what was learned`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 7. Personal / Errands / Shopping / Health / Social
  if (/buy|groceries|shopping|call|visit|meet|doctor|dentist|clean|tidy|pay|bills|bank|travel|pack|dinner|lunch/i.test(t)) {
    const summary = `Get this errand sorted out: ${title}.`;
    const steps = [
      `Check what is needed and make a quick plan`,
      `Head out or reach out at the scheduled time`,
      `Complete the errand and keep receipts/confirmations if needed`,
      `Cross it off the list`
    ];
    return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
  }

  // 8. Default / General Task
  const summary = `Work on ${title}. Take care of the key details and follow through step by step.`;
  const steps = [
    `Gather what is needed and plan a quick approach`,
    `Focus on the main work without distractions`,
    `Review everything once done`,
    `Mark task as completed`
  ];
  return `${summary}\n\nKey Steps:\n${steps.map(s => `- [ ] ${s}`).join('\n')}`;
}

/**
 * 2. Natural Language Task Parser: Extract structured task details with rich description, difficulty, and dates
 */
async function parseNaturalLanguageTask(options) {
  const text = typeof options === 'string' ? options : (options.text || '');
  const userDifficulty = typeof options === 'object' ? options.difficulty : null;
  const userDueDate = typeof options === 'object' ? options.dueDate : null;

  const ai = getGenAIClient();

  if (!ai) {
    return getMockNaturalLanguageParse({ text, difficulty: userDifficulty, dueDate: userDueDate });
  }

  const todayStr = new Date().toISOString().split('T')[0];

  const prompt = `You are a helpful, natural personal assistant scheduling a task.
The user wants to schedule a task from this raw input: "${text}"
${userDifficulty && userDifficulty !== 'Auto' ? `User requested priority: ${userDifficulty}` : ''}
${userDueDate ? `User requested due date: ${userDueDate}` : ''}
Today's Date: ${todayStr}

RULES:
1. Title: Crisp, short action title in Title Case (e.g. "Play Cricket", "Submit Docker Project", "Submit the Capstone Project").
   IMPORTANT: You MUST REMOVE scheduling words (e.g. "on 25th Nov", "tomorrow", "today", "by Friday"), priority words ("priority high", "urgent"), and status words ("in progress", "doing", "completed") from the title. NEVER include "on 25th Nov" or "priority high" in the title!
2. Description: Write a REAL, NATURAL, HUMAN description (1-2 friendly sentences explaining the task, followed by 3-4 simple, realistic steps).
   DO NOT use robotic corporate buzzwords like "execute end-to-end delivery", "focused technical precision", "high user experience standards", or "sanity verification". Keep it realistic, direct, and conversational, like a real person writing down their task.
3. Priority: "High" | "Medium" | "Low". (Casual activities like games/cricket default to Medium/Low unless marked urgent).
4. Status: "Pending" | "In Progress" | "Completed" (Default: "Pending").
5. Due Date: YYYY-MM-DD. Calculate accurately (e.g. "on 25th Nov" = ${new Date().getFullYear()}-11-25).

Respond ONLY with valid JSON (no markdown formatting, no code fences):
{
  "title": "Clean Action Title",
  "description": "Real human summary\n\nKey Steps:\n- [ ] Step 1\n- [ ] Step 2\n- [ ] Step 3",
  "priority": "High | Medium | Low",
  "status": "In Progress | Pending | Completed",
  "dueDate": "YYYY-MM-DD"
}
`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-flash-latest',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const responseText = response.text ? response.text.trim() : '';
    const cleaned = responseText.replace(/^```json\s*/i, '').replace(/```$/i, '').trim();
    const parsed = JSON.parse(cleaned);

    // Ensure title is strictly cleaned of metadata
    if (parsed.title) {
      parsed.title = cleanTaskTitle(parsed.title);
    }
    return parsed;
  } catch (err) {
    console.warn(`[GeminiService] Live API call failed (${err.message}). Using fallback parser.`);
    return getMockNaturalLanguageParse({ text, difficulty: userDifficulty, dueDate: userDueDate });
  }
}

/**
 * 3. Daily Focus Briefing: Summarize active tasks into a prioritized morning standup
 */
async function generateDailyBriefing(tasks = [], username = 'Developer') {
  const ai = getGenAIClient();

  if (!ai) {
    return getMockDailyBriefing(tasks, username);
  }

  const taskSummary = tasks.map((t, idx) => 
    `${idx + 1}. [${t.priority.toUpperCase()}] "${t.title}" (Status: ${t.status}, Due: ${t.dueDate ? t.dueDate.split('T')[0] : 'None'})`
  ).join('\n');

  const prompt = `You are an executive productivity coach conducting a daily morning standup briefing for ${username}.
Here are their current tasks:
${taskSummary || 'No active tasks at the moment.'}

Provide an inspiring, laser-focused daily plan.
Respond ONLY with valid JSON (no markdown formatting, no code fences):
{
  "greeting": "Warm greeting addressing ${username}",
  "headline": "Punchy 1-line theme for today",
  "topPriorities": [
    "Item 1: Crucial task to tackle first and why",
    "Item 2: Second priority",
    "Item 3: Third priority"
  ],
  "quickAdvice": "Actionable piece of productivity or engineering advice",
  "energyLevel": "High Focus | Steady Flow | High Urgency"
}`;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-flash-latest',
      contents: prompt,
      config: {
        responseMimeType: 'application/json'
      }
    });

    const responseText = response.text ? response.text.trim() : '';
    const cleaned = responseText.replace(/^```json\s*/i, '').replace(/```$/i, '').trim();
    return JSON.parse(cleaned);
  } catch (err) {
    console.warn(`[GeminiService] Live API call failed (${err.message}). Using fallback briefing.`);
    return getMockDailyBriefing(tasks, username);
  }
}

// ---------------------------------------------------------
// Intelligent Fallback Engines (Active when offline or without API Key)
// ---------------------------------------------------------

function getMockTaskBreakdown(title, description, priority) {
  const isCapstone = /capstone|thesis|dissertation|viva|rubric|faculty|submission|submit.*project|final.*project/i.test(title);
  const isDevOps = /docker|k8s|kubernetes|jenkins|pipeline|ci\/cd|terraform|ansible/i.test(title);
  const isBug = /fix|bug|error|issue|crash|patch/i.test(title);

  let subtasks = [];
  if (isCapstone) {
    subtasks = [
      'Synchronize git repository branches, clean up working tree, and tag production release',
      'Execute automated test suites (Jest/Supertest) and verify zero failing assertions',
      'Compile capstone project report, architecture diagrams, and milestone delivery summaries',
      'Dry-run live demonstration script covering REST API, AI features, Docker, and Kubernetes',
      'Package and submit project artifacts through the official evaluation portal'
    ];
  } else if (isDevOps) {
    subtasks = [
      'Define requirements, architecture, and environment configurations',
      'Create and test local configuration / manifests',
      'Integrate automated checks into the deployment pipeline',
      'Perform sanity testing and verify health check endpoints'
    ];
  } else if (isBug) {
    subtasks = [
      'Reproduce the issue locally with a failing test case',
      'Trace root cause in application logs and stack traces',
      'Implement the code patch and verify edge cases',
      'Run complete test suite and deploy fix'
    ];
  } else {
    subtasks = [
      'Clarify scope, objectives, and acceptance criteria',
      'Execute core implementation in focused intervals',
      'Review output against requirements and verify quality',
      'Mark deliverable complete and document key notes'
    ];
  }

  const cleanTitle = cleanTaskTitle(title);

  return {
    enhancedTitle: cleanTitle,
    summary: description && description.trim() 
      ? `Streamlined execution plan for "${cleanTitle}": ${description.slice(0, 100)}`
      : `High-impact execution breakdown for "${cleanTitle}" to ensure efficient delivery.`,
    suggestedPriority: priority || (isBug ? 'High' : 'Medium'),
    subtasks,
    estimatedTime: isCapstone ? '3-5 hours' : (isDevOps ? '2-4 hours' : (isBug ? '1-2 hours' : '1-3 hours')),
    acceptanceCriteria: `All subtasks are validated and functional without open regressions.`,
    proTip: `Group dependent items together and verify progress using your TaskFlow dashboard. (Simulated AI mode: add GEMINI_API_KEY in .env for live Gemini 2.5)`
  };
}

function getMockNaturalLanguageParse(input) {
  const text = typeof input === 'string' ? input : (input.text || '');
  const userDifficulty = typeof input === 'object' && input.difficulty ? input.difficulty : null;
  const userDueDate = typeof input === 'object' && input.dueDate ? input.dueDate : null;

  let raw = text.trim();

  // Normalize common typos and shortcuts before token extraction
  const normalized = raw
    .replace(/\b(tom+ar+ow|tom+or+ow|tom+or+o|tmrw|tmw)\b/gi, 'tomorrow')
    .replace(/\b(todai|2day)\b/gi, 'today')
    .replace(/\b(priorty|prio)\b/gi, 'priority')
    .replace(/\b(progres+|in-progress)\b/gi, 'in progress')
    .replace(/\b(pend?ing)\b/gi, 'pending')
    .replace(/\b(compl?eted?)\b/gi, 'completed');

  // 1. Difficulty / Priority determination
  let priority = userDifficulty && userDifficulty !== 'Auto' ? userDifficulty : 'Medium';
  if (!userDifficulty || userDifficulty === 'Auto') {
    if (/\b(?:priority\s*:?\s*(high|critical|urgent)|(high|critical|urgent)\s*priority)\b/i.test(normalized) ||
        /\b(urgent|critical|emergency|p0|p1)\b/i.test(normalized) ||
        /\b(k8s|kubernetes|cluster|prod|production|security|capstone)\b/i.test(normalized)) {
      priority = 'High';
    } else if (/\b(?:priority\s*:?\s*(low|minor)|(low|minor)\s*priority)\b/i.test(normalized) ||
               /\b(low|trivial|minor|whenever|readme|docs|comment|p3)\b/i.test(normalized)) {
      priority = 'Low';
    } else {
      priority = 'Medium';
    }
  }

  // 2. Status determination
  let status = 'Pending';
  if (/\b(?:status\s*:?\s*)?(in[- ]progress|in progress|wip|doing|working on|actively working|started)\b/i.test(normalized)) {
    status = 'In Progress';
  } else if (/\b(?:status\s*:?\s*)?(completed|done|finished|resolved)\b/i.test(normalized)) {
    status = 'Completed';
  }

  // 3. Due Date determination
  let dueDate = userDueDate || null;
  const today = new Date();
  if (!dueDate) {
    const explicitDate = parseExplicitDate(raw);
    if (explicitDate) {
      dueDate = explicitDate;
    } else if (/\b(today|tonight|eod|end of day)\b/i.test(normalized)) {
      dueDate = today.toISOString().split('T')[0];
    } else if (/\btomorrow\b/i.test(normalized)) {
      const tmrw = new Date(today);
      tmrw.setDate(tmrw.getDate() + 1);
      dueDate = tmrw.toISOString().split('T')[0];
    } else if (/\bday after tomorrow\b/i.test(normalized)) {
      const dayAfter = new Date(today);
      dayAfter.setDate(dayAfter.getDate() + 2);
      dueDate = dayAfter.toISOString().split('T')[0];
    } else if (/\b(?:by|on)?\s*friday\b/i.test(normalized)) {
      const fri = new Date(today);
      const day = fri.getDay();
      const diff = (5 - day + 7) % 7 || 7;
      fri.setDate(fri.getDate() + diff);
      dueDate = fri.toISOString().split('T')[0];
    } else if (/\b(?:by|on)?\s*monday\b/i.test(normalized)) {
      const mon = new Date(today);
      const day = mon.getDay();
      const diff = (1 - day + 7) % 7 || 7;
      mon.setDate(mon.getDate() + diff);
      dueDate = mon.toISOString().split('T')[0];
    } else {
      // Smart default based on priority: High = +1 day, Medium = +3 days, Low = +5 days
      const daysToAdd = priority === 'High' ? 1 : (priority === 'Medium' ? 3 : 5);
      const defaultDate = new Date(today);
      defaultDate.setDate(defaultDate.getDate() + daysToAdd);
      dueDate = defaultDate.toISOString().split('T')[0];
    }
  }

  // 4. Clean Action Title (extract clean title, stripping out priority, status, and date tokens)
  const title = cleanTaskTitle(raw);

  // 5. Generate clean, natural human description
  const formattedDescription = generateHumanDescription(title, raw);

  return {
    title,
    description: formattedDescription,
    priority,
    status,
    dueDate
  };
}

function getMockDailyBriefing(tasks = [], username = 'Developer') {
  const highPriority = tasks.filter(t => t.priority === 'High' && t.status !== 'Completed');
  const pending = tasks.filter(t => t.status === 'Pending');
  const inProgress = tasks.filter(t => t.status === 'In Progress');

  const topPriorities = [];
  if (highPriority.length > 0) {
    topPriorities.push(`⚡ High Priority: Focus on "${highPriority[0].title}" first.`);
  }
  if (inProgress.length > 0) {
    topPriorities.push(`🔄 In Progress: Push "${inProgress[0].title}" across the finish line.`);
  }
  if (pending.length > 0) {
    topPriorities.push(`📋 Plan ahead: Review "${pending[0].title}" before starting your next sprint.`);
  }
  if (topPriorities.length === 0) {
    topPriorities.push('✨ Great job! All current high-priority tasks are completed or organized.');
  }

  return {
    greeting: `Hello, ${username}! Here is your AI Daily Focus Plan.`,
    headline: highPriority.length > 0 ? 'Target High-Impact Deliverables First' : 'Maintain Steady Velocity & Momentum',
    topPriorities,
    quickAdvice: 'Work in 45-minute focused blocks. Tackle urgent tasks before checking emails or alerts.',
    energyLevel: highPriority.length > 0 ? 'High Urgency' : 'Steady Flow'
  };
}

module.exports = {
  breakdownTask,
  parseNaturalLanguageTask,
  generateDailyBriefing
};
