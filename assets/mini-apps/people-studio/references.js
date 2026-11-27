// Also imported by the local MCP server. Media origins follow the same split as the site build.
(function () {
  const libraries = {
    pose:['姿势','Pose'], depth:['深度姿势','Depth pose'], costume:['服装','Outfit'],
    anime:['动画服装','Anime outfit'], portrait:['人像','Portrait'], expr:['表情','Expression'],
    hair:['发型','Hairstyle'], dyn:['历代服饰','Historical dress'], scene:['场景','Scene'],
    light:['光影','Lighting'], prop:['道具','Prop'], furn:['家具','Furniture']
  };
  function reference(item, angle = 0, site, language = 'zh', assetHost = site) {
    const view = item.angles?.[angle];
    const image = view || item;
    const display = view ? image.i || image.m : image.m || image.i;
    const path = display || image.o || image.t;
    const url = image.sourceURL || new URL(path, (display ? site : assetHost).replace(/\/$/,'')+'/').href;
    const en = language === 'en';
    const label = [libraries[item.type][en ? 1 : 0], en ? item.id : item.ti || item.zh || item.id,
      view && (en ? view.en : view.zh)].filter(Boolean).join(' · ');
    return {label,url};
  }
  function formatBrief(prompt, references, language = 'zh') {
    const en = language === 'en';
    const lines = [];
    if (references.length) {
      lines.push(en ? 'Reference images' : '参考图');
      references.forEach((ref,index) => {
        const slot = (en ? 'Image ' : '图 ') + (index+1);
        const value = ref ? ref.label+' — '+ref.url : (en ? 'Attach this local image separately.' : '请另附这张本地参考图。');
        lines.push('- '+slot+' · '+value);
      });
    }
    return [prompt,lines.join('\n')].filter(Boolean).join('\n\n');
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports.reference = reference;
    module.exports.formatBrief = formatBrief;
  } else {
    window.PoseWearReferences = {reference,formatBrief};
  }
})();
