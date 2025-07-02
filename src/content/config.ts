import { defineCollection, z } from 'astro:content';

const projectsCollection = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    images: z.array(z.string()),
    tag: z.string(),
    techStack: z.array(z.string()),
    link: z.string().url().optional(),
    outputLink: z.string().url().optional(),
  }),
});

export const collections = {
  'projects': projectsCollection,
};