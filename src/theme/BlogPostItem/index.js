import React from 'react';
import BlogPostItem from '@theme-original/BlogPostItem';
import Comment from '../../components/comments/Comments';

export default function BlogPostItemWrapper(props) {
  return (
    <>
      <BlogPostItem {...props} />
      <Comment />
    </>
  );
}
